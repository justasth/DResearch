"""Python 3, standard library only. Run: python3 reproduce.py
Reproduces statistical tables from the released CSVs, not automatic semantic coding.
"""
import csv, collections, hashlib, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / '复现统计'
OUT.mkdir(exist_ok=True)
def read(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))
def save(name, header, rows):
    with open(OUT/name, 'w', encoding='utf-8-sig', newline='') as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)
raw=read(ROOT/'rawdata.csv'); r=read(ROOT/'results.csv')
towns=['朱家角','新场','枫泾','南翔','川沙','金泽','高桥','练塘','嘉定','罗店','张堰']
dims=['历史文化与遗产','空间与环境','日常生活与社会交往','商业与游憩活动','交通与公共服务']
assert len({x['语义单元ID'] for x in r})==len(r)
rk={(x['古镇'],x['笔记ID']) for x in raw}
assert len(rk)==len(raw)
assert all((x['古镇'],x['笔记ID']) in rk for x in r)
assert all(x['感知维度'] in dims for x in r)
assert all(x['评价倾向'] in ['正面','负面','褒贬并存','不明确'] for x in r)
rc=collections.Counter(x['古镇'] for x in raw)
count=collections.Counter((x['古镇'],x['具体对象']) for x in r)
selected=[x for x in r if count[x['古镇'],x['具体对象']]>=2]
sc=collections.Counter(x['古镇'] for x in selected)
full=collections.Counter(x['古镇'] for x in r)
save('各镇样本与语义数量.csv',['古镇','笔记记录数','全量语义记录数','具体对象镇内频次≥2语义记录数'],
     [[t,rc[t],full[t],sc[t]] for t in towns])
for name, rows in [('全量',r),('图4',selected)]:
    c=collections.Counter((x['古镇'],x['感知维度']) for x in rows)
    save(name+'_五类数量.csv',['古镇']+dims+['合计'],
         [[t]+[c[t,d] for d in dims]+[sum(c[t,d] for d in dims)] for t in sorted(towns,key=lambda t:-sum(c[t,d] for d in dims))])
presence=collections.defaultdict(set)
for x in r:presence[x['具体对象']].add(x['古镇'])
objdim=collections.defaultdict(set)
for x in r:objdim[x['古镇'],x['具体对象']].add(x['感知维度'])
save('图5_词云对象与覆盖.csv',['古镇','具体对象','频次','感知维度','覆盖镇数','标记'],
     [[t,o,n,'；'.join(sorted(objdim[t,o])),len(presence[o]),'*' if len(presence[o])==1 else '†' if len(presence[o])==11 else '']
      for (t,o),n in sorted(count.items(),key=lambda kv:(towns.index(kv[0][0]),-kv[1],kv[0][1])) if n>=2])
save('图6_对象节点候选.csv',['古镇','具体对象','频次','覆盖镇数'],
     [[t,o,n,len(presence[o])] for (t,o),n in sorted(count.items()) if n>2 and len(presence[o]) in (1,11)])
byid={x['语义单元ID']:x for x in r}
emotion_rows=read(HERE/'图6情感关联纳入清单.csv')
ec=collections.Counter()
for x in emotion_rows:
    y=byid[x['语义单元ID']]
    assert all(x[k]==y[k] for k in ['古镇','具体对象','情感类别'])
    ec[x['古镇'],x['具体对象'],x['情感类别']]+=1
save('图6_情感节点.csv',['古镇','具体对象','情感类别','关联记录数'],[list(k)+[n] for k,n in sorted(ec.items())])
pol=collections.Counter((x['古镇'],x['评价倾向']) for x in r)
save('全量评价数量.csv',['古镇','正面','负面','褒贬并存','不明确'],
     [[t]+[pol[t,k] for k in ['正面','负面','褒贬并存','不明确']] for t in towns])
mapping={x['语义单元ID']:x for x in read(HERE/'评价主题对应.csv')}
a=[]
for x in r:
    if x['评价倾向'] in ('正面','负面'):
        assert x['语义单元ID'] in mapping
        a.append(dict(x,**mapping[x['语义单元ID']]))
freq=collections.Counter((x['古镇'],x['评价倾向'],x['归并主题']) for x in a)
a=[x for x in a if freq[x['古镇'],x['评价倾向'],x['归并主题']]>=2]
for x in a:
    x['显示维度']='其他正面' if x['展示主题'].startswith('其他正面') else x['感知维度']
node=collections.Counter((x['评价倾向'],x['显示维度'],x['展示主题']) for x in a)
a=[x for x in a if node[x['评价倾向'],x['显示维度'],x['展示主题']]>=2]
den=collections.Counter((x['古镇'],x['评价倾向']) for x in a)
flow=collections.Counter((x['古镇'],x['评价倾向'],x['显示维度'],x['展示主题']) for x in a)
save('图7_流量与占比.csv',['古镇','评价倾向','显示维度','评价主题','记录数','镇内同方向入图数','镇内占比'],
     [list(k)+[n,den[k[0],k[1]],n/den[k[0],k[1]]] for k,n in sorted(flow.items())])
save('图7_纳入语义ID.csv',['语义单元ID'],[[x['语义单元ID']] for x in a])
assert len(selected)==12128
assert collections.Counter(x['评价倾向'] for x in a)=={'正面':4331,'负面':376}
manifest={'raw_records':len(raw),'unique_notes':len({x['笔记ID'] for x in raw}),
 'semantic_records':len(r),'notes_with_semantics':len({x['笔记ID'] for x in r}),
 'figure4_records':len(selected),'figure7_records':dict(collections.Counter(x['评价倾向'] for x in a)),
 'sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['rawdata.csv','results.csv']}}
(OUT/'verification.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(manifest,ensure_ascii=False,indent=2))
