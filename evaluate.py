"""Evaluate speech-only predictions against the visually verified Unity proxy; plot and test results.

Commands:
  python evaluate.py proxy --run outputs/demo --output-dir outputs/proxy_report
  python evaluate.py batch --data-root Data --participant P1 --limit 1 --output-root outputs/p1_pilot
  python evaluate.py plot --report outputs/proxy_report/results.json --output outputs/comparison.png
  python evaluate.py test

Scores use simplified V2 geometry, not verified physical 3D accuracy.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
import re
import tempfile
import unittest


# ================== PROXY ACCURACY ==================
SIZES = {'red': (2.0, 4.0, 2.0), 'blue': (4.0, 1.0, 2.0), 'green': (2.0, 1.0, 4.0)}


SCORING_FIELDS = ['coverage','shape_correct','color_f1','xz_f1','xyz_f1','oriented_f1','full_tower_exact']


OUTPUT_FIELDS = ['run','participant','trial','step','mode','source','tower_id','status','prediction_available',
                 'shape_predicted','shape_target','coverage','shape_correct','color_f1','xz_f1',
                 'xyz_f1','oriented_f1','full_tower_exact','predicted_block_count','target_block_count','error']


def read_json(path: Path | str) -> dict:
    with Path(path).open(encoding='utf-8') as f:
        return json.load(f)


def get_target_metadata(run: Path) -> tuple[str | None, dict]:
    for name in ('dataset_input.json', 'live_log.json'):
        p = run/name
        if p.is_file():
            d=read_json(p)
            return d.get('target_for_local_log_only',{}).get('tower_id'), {
                'participant':d.get('participant'), 'trial':d.get('trial'),
                'step':d.get('step', d.get('steps',[{}])[0].get('step_num') if d.get('steps') else None),
                'source_meta':name,
            }
    return None, {}


def get_run_prediction(run: Path) -> tuple[dict | None, str, str, str | None]:
    """Return tower, inference provenance, status, and optional predicted shape."""
    live=run/'live_log.json'
    if live.is_file():
        log=read_json(live)
        if log.get('source') not in ('gemini','mock'):
            raise ValueError('Unknown live provenance; cannot verify inference source')
        tower=read_json(run/'final_tower.json') if (run/'final_tower.json').is_file() else None
        events=log.get('events',[])
        status=(events[-1].get('status') if events else 'no_prediction')
        shape=(log['requests'][-1].get('prediction',{}).get('shape') if log.get('requests') else None)
        return tower, log.get('source'), status, shape
    towerpath=run/'tower.json'
    predpath=run/'prediction.json'
    timelinepath=run/'timeline.json'
    if predpath.is_file():
        p=read_json(predpath)
        source=p.get('source','unknown')
        status=p.get('prediction',{}).get('status','unknown')
        shape=p.get('prediction',{}).get('shape')
    elif timelinepath.is_file():
        p=read_json(timelinepath)
        source=p.get('source','unknown')
        status=p.get('plan',{}).get('status','unknown')
        shape=p.get('plan',{}).get('shape')
    else:
        source='unknown'; status='unknown'; shape=None
    return (read_json(towerpath) if towerpath.is_file() else None,source,status,shape)


def coordinates(block: dict) -> tuple[float,float,float]:
    color=block['color'];axis=block.get('axis')
    if color not in SIZES: raise ValueError('Unknown block color: '+str(color))
    if axis not in ('x','y','z'): raise ValueError('Unknown block axis: '+str(axis))
    # Match V2's axis convention for red and horizontal blue/green.
    if color=='red' and axis!='y':raise ValueError('Red block must have axis=y')
    dx,dy,dz = (SIZES[color] if axis=={'red':'y','blue':'x','green':'z'}[color]
                else ((2.0,1.0,4.0) if axis=='z' else (4.0,1.0,2.0)))
    return (float(block['x'])+dx/2, float(block['y'])+dy/2, float(block['z'])+dz/2)


def oriented(block:dict)->str:
    return block['axis']


def prf(matched:int, expected:int, predicted:int)->tuple[float,float,float]:
    prec=matched/predicted if predicted else 0.0
    rec=matched/expected if expected else 0.0
    return prec,rec,2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0


def matching(expected:list, predicted:list, criterion)->tuple[int,list]:
    """Maximum one-to-one matching; repeated colors are not double-counted."""
    if len(predicted)>30:
        raise ValueError('Unreasonably large predicted scene (>30 blocks)')
    @lru_cache(maxsize=None)
    def dfs(i:int, used:int):
        if i==len(expected):return (0,())
        best=dfs(i+1,used)
        for j,p in enumerate(predicted):
            if used & (1<<j) or not criterion(expected[i],p):continue
            n,pairs=dfs(i+1,used|(1<<j))
            if n+1>best[0]:best=(n+1,((i,j),)+pairs)
        return best
    count,pairs=dfs(0,0)
    return count,[{'target_index':i,'prediction_index':j} for i,j in pairs]


def block_scores(target_blocks:list, pred_blocks:list, tol:float=0.25)->dict:
    for b in target_blocks+pred_blocks:coordinates(b)
    center={id(b):coordinates(b) for b in target_blocks+pred_blocks}
    def distance(a,b,axes):
        return all(abs(center[id(a)][k]-center[id(b)][k])<=tol+1e-9 for k in axes)
    methods={
        'color':lambda a,b: a['color']==b['color'],
        'xz': lambda a,b:a['color']==b['color'] and distance(a,b,(0,2)),
        'xyz': lambda a,b:a['color']==b['color'] and distance(a,b,(0,1,2)),
        'oriented':lambda a,b:a['color']==b['color'] and distance(a,b,(0,1,2)) and oriented(a)==oriented(b),
    }
    out={}
    for name,fn in methods.items():
        n,pairs=matching(target_blocks,pred_blocks,fn)
        precision,recall,f1=prf(n,len(target_blocks),len(pred_blocks))
        out[name+'_matched']=n
        out[name+'_precision']=round(precision,4)
        out[name+'_recall']=round(recall,4)
        out[name+'_f1']=round(f1,4)
        out[name+'_pairs']=pairs
    out['full_tower_exact']=int(bool(target_blocks) and len(target_blocks)==len(pred_blocks)
                                and out['oriented_matched']==len(target_blocks))
    return out


def flip_block(block:dict,flip_x:bool,flip_z:bool)->dict:
    o=dict(block)
    ax=o['axis'];color=o['color']
    if ax=='x':dx,dz=4,2
    elif ax=='z':dx,dz=2,4
    else:dx,dz=2,2
    if flip_x:o['x']=4-float(o['x'])-dx
    if flip_z:o['z']=4-float(o['z'])-dz
    return o


def validate_target_id(target_id:str, refs:dict)->dict:
    if not target_id or not isinstance(target_id,str):
        raise ValueError('Missing canonical target ID: expected tower_list.txt or explicit --target')
    if target_id not in refs:
        raise ValueError(f'Unrecognized target {target_id!r}; expected full Unity ID, e.g. t-8-c1-translucent')
    return refs[target_id]


def evaluate(run:Path, refs:dict, *, target_id:str|None=None, tol:float=0.25,
             flip_x:bool=False,flip_z:bool=False,allow_nonllm:bool=False)->dict:
    found,metadata=get_target_metadata(run)
    if target_id is not None and found is not None and target_id !=found:
        raise ValueError(f'Explicit target {target_id!r} disagrees with canonical run target {found!r}')
    tgt=validate_target_id(target_id or found,refs)
    pred,source,status,shape=get_run_prediction(run)
    if source!='live_llm' and source!='gemini' and not allow_nonllm:
        raise ValueError(f'{run.name}: prediction source {source!r} is not verified live Gemini; use --allow-nonllm only to test with fixtures')
    actual=[flip_block({k:b[k] for k in ('color','axis','x','y','z')},flip_x,flip_z)
            for b in tgt['blocks']]
    blocks=pred.get('blocks',[]) if pred else []
    if blocks is None:blocks=[]
    if not isinstance(blocks,list):raise ValueError('Invalid predicted blocks list')
    metrics=block_scores(actual,blocks,tol)
    filled=bool(pred and len(blocks)>0)
    if pred and pred.get('shape') is not None:shape=pred['shape']
    result={'run':str(run),'participant':metadata.get('participant'), 'trial':metadata.get('trial'),
            'step':metadata.get('step'),'mode':'live' if (run/'live_log.json').exists() else 'offline',
            'source':source,'tower_id':tgt['tower_id'],'status':status,
            'prediction_available':int(filled), 'shape_predicted':shape,'shape_target':tgt['shape'],
            'coverage':int(filled),'shape_correct':int(shape==tgt['shape']) if shape else None,
            'target_block_count':len(actual),'predicted_block_count':len(blocks),
            'color_f1':metrics['color_f1'],'xz_f1':metrics['xz_f1'],
            'xyz_f1':metrics['xyz_f1'],'oriented_f1':metrics['oriented_f1'],
            'full_tower_exact':metrics['full_tower_exact'],'error':None,
            'matched_details':metrics,'target_blocks':actual,'predicted_blocks':blocks,
            'calibration':'PROVISIONAL_UNITY_TO_V2_PROXY_NOT_PHYSICAL_ACCURACY',
            'tolerance_v2_units':tol,'flip_x':flip_x,'flip_z':flip_z}
    if (run/'live_log.json').exists():
        log=read_json(run/'live_log.json')
        result['timeline_scores']=[{'applied_audio_clock':event['applied_audio_clock'],
          'speech_cutoff':event.get('speech_cutoff'),
          'status':event['status'],
          **{key:block_scores(actual,event['blocks'],tol)[key] for key in ('color_f1','xz_f1','xyz_f1','oriented_f1','full_tower_exact')}}
          for event in log.get('events',[])]
    return result


def aggregate(results:list)->dict:
    out={'count':len(results),'warning':'Provisional V2 normalized proxies, not verified physical 3D accuracy'}
    for metric in SCORING_FIELDS:
        nums=[r[metric] for r in results if r[metric] is not None]
        out[metric+'_mean']=round(sum(nums)/len(nums),4) if nums else None
        out[metric+'_denominator']=len(nums)
    out['ambiguous_or_no_tower']=sum(1 for r in results if not r['prediction_available'])
    return out


def main_proxy():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,action='append', help='One output directory; repeat for multiple runs')
    p.add_argument('--runs-root',type=Path,help='Find V2 output directories underneath this directory')
    p.add_argument('--target',help='Full tower ID; allowed only when evaluating one run')
    p.add_argument('--reference',type=Path,default=Path(__file__).resolve().parent / 'unity_target_reference.json')
    p.add_argument('--output-dir',type=Path,default=Path('accuracy_report'))
    p.add_argument('--tolerance',type=float,default=0.25)
    p.add_argument('--flip-x',action='store_true',help='Mirror reference X; choose once after calibration, not separately per trial')
    p.add_argument('--flip-z',action='store_true',help='Mirror reference Z; choose once after calibration, not separately per trial')
    p.add_argument('--allow-nonllm',action='store_true',help='Permit mock/fixture runs strictly for local testing')
    args=p.parse_args()
    if args.tolerance<=0:p.error('Tolerance must be positive')
    runs=list(args.run or [])
    if args.runs_root:
        root=args.runs_root
        if not root.is_dir():p.error('Not a directory: '+str(root))
        seen=set()
        for x in list(root.rglob('dataset_input.json'))+list(root.rglob('live_log.json')):
            if x.parent not in seen:runs.append(x.parent);seen.add(x.parent)
    runs=list(dict.fromkeys(r.resolve() for r in runs))
    if not runs:p.error('Use --run OUTPUT_FOLDER or --runs-root PARENT_FOLDER')
    if args.target and len(runs)!=1:p.error('--target can only be used with a single --run')
    ref=read_json(args.reference)
    if ref.get('schema')!='gta-unity-v2-proxy-reference-v1':p.error('Unexpected reference schema')
    refs={t['tower_id']:t for t in ref['towers']}
    if len(refs)!=12:p.error('Expected 12 unique Unity target towers')
    results=[];failures=[]
    for run in runs:
        try:
            if not run.is_dir():raise FileNotFoundError(run)
            result=evaluate(run,refs,target_id=args.target,tol=args.tolerance,
                flip_x=args.flip_x,flip_z=args.flip_z,allow_nonllm=args.allow_nonllm)
            results.append(result)
            print(f"{run.name}: target {result['tower_id']} | full tower {result['full_tower_exact']} | "
                  f"oriented F1 {result['oriented_f1']:.2f} | xz F1 {result['xz_f1']:.2f} | "
                  f"color F1 {result['color_f1']:.2f}")
        except (OSError,ValueError,KeyError,TypeError) as e:
            failures.append({'run':str(run),'error':str(e)})
            print(f'SKIPPED {run}: {e}')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with (args.output_dir/'per_trial.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=OUTPUT_FIELDS);wr.writeheader()
        for r in results:wr.writerow({k:r.get(k) for k in OUTPUT_FIELDS})
    summary=aggregate(results)
    (args.output_dir/'results.json').write_text(json.dumps({'summary':summary,'trials':results,'skipped':failures},indent=2))
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2))
    print('\nSUMMARY:',json.dumps(summary,indent=2))
    print('Reports:',args.output_dir.resolve())
    if failures:print('WARNING:',len(failures),'runs were skipped; do not use the summary denominator as total attempted trials')
    if not results:raise SystemExit('No eligible runs were scored. Check target IDs and prediction source.')

# ================== SIDE BY SIDE VISUALIZATION ==================
def dimensions(b):
    if b['color']=='red':return (2,4,2)
    return (4,1,2) if b['axis']=='x' else (2,1,4)


def draw(ax,blocks,title):
    for block in blocks:
        dx,dy,dz=dimensions(block)
        ax.bar3d(block['x'],block['z'],block['y'],dx,dz,dy,
                 color=block['color'],edgecolor='black',alpha=0.92,shade=True)
    for v in (0,2,4):
        ax.plot([v,v],[0,4],[0,0],color='#777777',lw=1.5)
        ax.plot([0,4],[v,v],[0,0],color='#777777',lw=1.5)
    for x,z,label in ((1,1,'back-left'),(3,1,'back-right'),(1,3,'front-left'),(3,3,'front-right')):
        ax.text(x,z,0,label,fontsize=7,color='#555555',ha='center')
    ax.set(xlim=(-2,6),ylim=(6,-2),zlim=(0,10))
    ax.view_init(elev=28,azim=-62)
    ax.set_box_aspect((8,8,10))
    ax.set_axis_off()
    ax.set_title(title,fontsize=11)


def plot_results(report_path:Path,output:Path,index:int):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=json.loads(report_path.read_text(encoding='utf8'))
    trials=report['trials']
    if index<0 or index>=len(trials):raise IndexError(f'Index {index} not in 0..{len(trials)-1}')
    r=trials[index]
    fig=plt.figure(figsize=(12,5.8))
    fig.suptitle(f"Provisional Unity proxy vs V2 prediction | {r['tower_id']}\n"
                 f"Exact: {r['full_tower_exact']} | XZ F1: {r['xz_f1']} | Orientation F1: {r['oriented_f1']}",fontsize=12)
    ax1=fig.add_subplot(121,projection='3d'); ax2=fig.add_subplot(122,projection='3d')
    draw(ax1,r['target_blocks'],'Unity-derived target (mapped to V2)')
    draw(ax2,r['predicted_blocks'],f"{r['mode']} predicted tower ({r['status']})")
    fig.text(0.02,0.025,'Viewer-relative axes/height are not independently mesh-calibrated; do not use as physical-accuracy evidence.',fontsize=9)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=140,bbox_inches='tight');plt.close(fig)
    print('Saved',output)


def main_plot():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True,help='results.json from evaluate_accuracy.py')
    p.add_argument('--index',type=int,default=0,help='0-based trial index in the report')
    p.add_argument('--output',type=Path,default=Path('tower_comparison.png'))
    a=p.parse_args()
    plot_results(a.report,a.output,a.index)

# ================== REGRESSION TESTS ==================
HERE=Path(__file__).resolve().parent


REFS={t['tower_id']:t for t in read_json(HERE/'unity_target_reference.json')['towers']}


def strip(target):
    return [{k:b[k] for k in ('color','axis','x','y','z')} for b in target['blocks']]


class EvaluationTests(unittest.TestCase):
    def test_all_12_targets_are_exact_when_prediction_matches(self):
        self.assertEqual(len(REFS),12)
        for target in REFS.values():
            b=strip(target)
            s=block_scores(b,b)
            self.assertEqual(s['full_tower_exact'],1,target['tower_id'])
            self.assertEqual(s['color_f1'],1)
            self.assertEqual(s['oriented_f1'],1)

    def test_same_colors_but_wrong_position_is_not_exact(self):
        c1=strip(REFS['t-8-c1-translucent'])
        c3=strip(REFS['t-8-c3-translucent'])
        s=block_scores(c1,c3)
        self.assertEqual(s['color_f1'],1.0)
        self.assertEqual(s['full_tower_exact'],0)
        self.assertLess(s['xz_f1'],1)

    def test_orientations_detected(self):
        tr=strip(REFS['t-8-tr1-translucent'])
        altered=[dict(x) for x in tr]
        altered[2]['axis']='z'
        altered[2]['x']+=1
        altered[2]['z']-=1
        s=block_scores(tr,altered)
        self.assertEqual(s['xyz_f1'],1.0) # center unchanged, but axis differs
        self.assertLess(s['oriented_f1'],1)

    def test_duplicates_matched_once(self):
        l=strip(REFS['t-8-l1-translucent'])
        s=block_scores(l,l[:-1])
        self.assertEqual(s['color_matched'],2)
        self.assertAlmostEqual(s['color_f1'],0.8)
        self.assertFalse(s['full_tower_exact'])

    def test_run_metadata_target_and_shape(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);target='t-8-c2-translucent'
            (p/'dataset_input.json').write_text(json.dumps({'participant':'P1','trial':1,'steps':[{'step_num':1}],
                'target_for_local_log_only':{'tower_id':target}}))
            (p/'tower.json').write_text(json.dumps({'schema':'gta-world-v2','shape':'C','blocks':strip(REFS[target])}))
            (p/'prediction.json').write_text(json.dumps({'source':'live_llm','prediction':{'status':'ready','shape':'C'}}))
            r=evaluate(p,REFS)
            self.assertEqual(r['full_tower_exact'],1)
            self.assertEqual(r['shape_correct'],1)
            self.assertEqual(r['participant'],'P1')

    def test_ambiguous_output_counted_as_zero(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            (p/'dataset_input.json').write_text(json.dumps({'target_for_local_log_only':{'tower_id':'t-8-c1-translucent'}}))
            (p/'prediction.json').write_text(json.dumps({'source':'live_llm','prediction':{'status':'ambiguous','shape':'C'}}))
            r=evaluate(p,REFS)
            self.assertEqual(r['coverage'],0)
            self.assertEqual(r['full_tower_exact'],0)
            self.assertEqual(aggregate([r])['count'],1)

    def test_mock_rejected_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            (p/'live_log.json').write_text(json.dumps({'source':'mock','events':[],
                'target_for_local_log_only':{'tower_id':'t-8-l1-translucent'}}))
            with self.assertRaisesRegex(ValueError,'not verified live Gemini'):
                evaluate(p,REFS)

    def test_target_override_disagreement_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            (p/'dataset_input.json').write_text(json.dumps({'target_for_local_log_only':{'tower_id':'t-8-c1-translucent'}}))
            with self.assertRaisesRegex(ValueError,'disagrees'):
                evaluate(p,REFS,target_id='t-8-l1-translucent')


class ReferenceIntegrityTests(unittest.TestCase):
    def test_reference_ids_and_block_count(self):
        reference = read_json(HERE / 'unity_target_reference.json')
        self.assertEqual(reference['schema'], 'gta-unity-v2-proxy-reference-v1')
        self.assertEqual(len(reference['towers']), 12)
        ids = set()
        for tower in reference['towers']:
            tid = tower['tower_id']
            self.assertNotIn(tid, ids)
            ids.add(tid)
            suffix = tid.split('-')[2]
            shape = ('TREE' if suffix.startswith('tr') else suffix[0].upper())
            self.assertEqual(tower['shape'], shape)
            self.assertEqual(len(tower['blocks']), 3)

    def test_known_corrected_front_back_positions(self):
        # L1 sits in the near/front row of the viewer-relative grid after Z correction.
        l1 = REFS['t-8-l1-translucent']
        blue = next(b for b in l1['blocks'] if b['color'] == 'blue')
        self.assertAlmostEqual(blue['z'], 2.0)



def main_batch():
    """Make real Vertex calls for a participant, evaluate, save a trial CSV + summary."""
    from dataset import inventory, load_selection
    from tower import DEFAULT_MODEL, run as run_prediction

    p = argparse.ArgumentParser(description="Run/evaluate speech-only Vertex Gemini on participant trials")
    p.add_argument('--data-root', required=True, type=Path, help='Original GTA Data/ folder containing P1')
    p.add_argument('--participant', default='P1')
    p.add_argument('--trial', type=int, help='Only this trial (otherwise all available)')
    p.add_argument('--limit', type=int, help='Run only first N trials (pilot with --limit 1)')
    p.add_argument('--selection', choices=('full-trial','first-step'), default='full-trial',
                   help='Use all messages in each trial (default) or just step 1')
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--output-root', type=Path, default=Path('outputs/p1_vertex_full_trial'))
    p.add_argument('--resume', action='store_true', help='Reuse saved predictions without new API calls')
    p.add_argument('--dry-run', action='store_true', help='Show selected trials without calling Gemini')
    p.add_argument('--tolerance', type=float, default=0.25)
    args = p.parse_args()
    if args.limit is not None and args.limit < 1:
        p.error('--limit must be positive')
    if args.tolerance <= 0:
        p.error('--tolerance must be positive')
    chosen = [row for row in inventory(args.data_root)
              if row['participant'].lower() == args.participant.lower()
              and (args.trial is None or row['trial'] == args.trial)]
    if args.limit is not None:
        chosen = chosen[:args.limit]
    if not chosen:
        p.error('No trials found for participant/trial in the supplied Data folder')
    refs = REFS
    rows = []
    skipped = []
    print(f"Participant {args.participant} | {len(chosen)} trials | selection={args.selection} | model={args.model}")
    for entry in chosen:
        trial = entry['trial']
        name = f"{entry['participant']}_T{trial:02d}_{args.selection.replace('-','_')}"
        run_dir = args.output_root / 'runs' / name
        try:
            if not entry['steps']:
                raise ValueError('Trial has no recorded steps')
            if args.selection == 'full-trial':
                record = load_selection(args.data_root, entry['participant'], trial,
                                        through_step=max(entry['steps']))
            else:
                record = load_selection(args.data_root, entry['participant'], trial, step=1)
            tid = record['target_for_local_log_only']['tower_id']
            if tid not in refs:
                raise ValueError(f'Canonical target is absent/unknown: {tid!r}; no API call made')
            if args.dry_run:
                print(f"DRY RUN {name} | {len(record['steps'])} step(s) | {tid}")
                continue
            if run_dir.exists():
                if not args.resume:
                    raise FileExistsError(f'{run_dir} exists. Choose a fresh --output-root or use --resume')
                pred_path = run_dir / 'prediction.json'
                data_path = run_dir / 'dataset_input.json'
                if not pred_path.exists() or not data_path.exists():
                    raise ValueError('Existing run is incomplete; not reusing it')
                old_pred = read_json(pred_path)
                old_data = read_json(data_path)
                if (old_pred.get('model') != args.model or
                    old_data.get('model_input') != record['model_input'] or
                    old_pred.get('source') != 'live_llm'):
                    raise ValueError('Existing run does not match model / transcript / live provenance')
                print(f"REUSE {name} (no new Gemini call)")
            else:
                print(f"GEMINI {name}: sending speech only ({len(record['model_input'])} characters)", flush=True)
                run_prediction(record, run_dir, model=args.model)
            result = evaluate(run_dir, refs, tol=args.tolerance)
            rows.append(result)
            print(f"RESULT {name}: target={tid} | exact={result['full_tower_exact']} | "
                  f"oriented_F1={result['oriented_f1']:.2f} | color_F1={result['color_f1']:.2f} "
                  f"| status={result['status']}", flush=True)
        except Exception as exc:
            skipped.append({'trial':trial,'run':str(run_dir),'error':str(exc)})
            print(f"FAILED {name}: {exc}", flush=True)
    if args.dry_run:
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / 'per_trial.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({field:r.get(field) for field in OUTPUT_FIELDS})
    summary = aggregate(rows)
    summary.update({'participant':args.participant,'model':args.model,'selection':args.selection,
                    'attempted':len(chosen),'scored':len(rows),'failed':len(skipped),
                    'strict_note':'Full-tower exact is within the fixed V2 coordinate tolerance, not pixel accuracy'})
    (args.output_root / 'summary.json').write_text(json.dumps(summary,indent=2))
    (args.output_root / 'results.json').write_text(json.dumps({'summary':summary,'trials':rows,'skipped':skipped},indent=2))
    print('\nFINAL P1 RESULTS')
    print('Trials attempted:',len(chosen),'| Scored:',len(rows),'| Failed:',len(skipped))
    print('Full-tower exact:',summary['full_tower_exact_mean'])
    print('Color F1:',summary['color_f1_mean'],'| XZ F1:',summary['xz_f1_mean'],
          '| XYZ F1:',summary['xyz_f1_mean'],'| Oriented F1:',summary['oriented_f1_mean'])
    print('Reports:',args.output_root.resolve())
    if skipped:
        print('IMPORTANT: Failed API/invalid-target trials are listed in results.json/skipped and not scored.')
    if not rows:
        raise SystemExit('No eligible real Gemini predictions were evaluated. See results.json/skipped.')


def main():
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('batch','proxy','plot','test'))
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    if args.command == 'batch':
        main_batch()
    elif args.command == 'proxy':
        main_proxy()
    elif args.command == 'plot':
        main_plot()
    else:
        suite = unittest.TestSuite([
            unittest.defaultTestLoader.loadTestsFromTestCase(EvaluationTests),
            unittest.defaultTestLoader.loadTestsFromTestCase(ReferenceIntegrityTests),
        ])
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
