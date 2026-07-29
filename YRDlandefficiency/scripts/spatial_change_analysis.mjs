import fs from 'node:fs/promises';

import { fileURLToPath } from 'node:url';
const root=fileURLToPath(new URL('..',import.meta.url));
const codes=[310000,320000,330000,340000],features=[];
for(const code of codes){const suffix=code===310000?'':'_full';const j=JSON.parse(await fs.readFile(`${root}/data/boundaries/${code}${suffix}.json`,'utf8'));for(const f of j.features)features.push({code,...f});}
const metrics=JSON.parse(await fs.readFile(`${root}/results/derived/computed_metrics.json`,'utf8'));
const norm=n=>String(n).replace(/市$/u,'');
function pts(g){return(g.type==='Polygon'?g.coordinates:g.coordinates.flat()).flat()}
const vertex=new Map();
for(const f of features){const n=f.code===310000?'上海':norm(f.properties.name);if(!vertex.has(n))vertex.set(n,new Set());for(const[x,y]of pts(f.geometry))vertex.get(n).add(`${x.toFixed(4)},${y.toFixed(4)}`)}
const names=[...new Set(metrics.rows.map(r=>norm(r.city)))].sort((a,b)=>a.localeCompare(b,'zh-CN'));
const adj=new Map(names.map(n=>[n,new Set()]));
for(let i=0;i<names.length;i++)for(let j=i+1;j<names.length;j++){const a=vertex.get(names[i]),b=vertex.get(names[j]);let hit=false;for(const q of a)if(b.has(q)){hit=true;break}if(hit){adj.get(names[i]).add(names[j]);adj.get(names[j]).add(names[i])}}
// Island correction: preserve the maritime Shanghai–Zhoushan Queen link and
// add the nearest mainland connection between Zhoushan and Ningbo.
adj.get('舟山').add('宁波');adj.get('宁波').add('舟山');
if(names.some(n=>adj.get(n).size===0))throw new Error(`Isolated cities: ${names.filter(n=>adj.get(n).size===0)}`);
const idx=new Map(names.map((n,i)=>[n,i]));
const W=names.map(n=>{const row=Array(names.length).fill(0),d=adj.get(n).size;for(const q of adj.get(n))row[idx.get(q)]=1/d;return row});
const lag=v=>W.map(row=>row.reduce((s,w,j)=>s+w*v[j],0));
function moran(vals){const m=vals.reduce((a,b)=>a+b,0)/vals.length,z=vals.map(v=>v-m),l=lag(z);return z.reduce((s,v,i)=>s+v*l[i],0)/z.reduce((s,v)=>s+v*v,0)}
function rng(seed){return()=>{seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function shuffle(a,random){const x=[...a];for(let i=x.length-1;i>0;i--){const j=Math.floor(random()*(i+1));[x[i],x[j]]=[x[j],x[i]]}return x}
function quantile(a,p){const x=[...a].sort((u,v)=>u-v),q=(x.length-1)*p,l=Math.floor(q),h=Math.ceil(q);return x[l]+(x[h]-x[l])*(q-l)}
function globalTest(vals,seed){const I=moran(vals),random=rng(seed),perms=[];for(let k=0;k<999;k++)perms.push(moran(shuffle(vals,random)));const p=(1+perms.filter(x=>Math.abs(x)>=Math.abs(I)).length)/1000;return{I,p,lo:quantile(perms,.025),hi:quantile(perms,.975),perm_mean:perms.reduce((a,b)=>a+b,0)/perms.length}}
function lisa(vals,seed){const n=vals.length,m=vals.reduce((a,b)=>a+b,0)/n,z=vals.map(v=>v-m),m2=z.reduce((s,v)=>s+v*v,0)/n,l=lag(z),random=rng(seed);return names.map((city,i)=>{const obs=z[i]*l[i]/m2,k=adj.get(city).size,others=z.filter((_,j)=>j!==i),sim=[];for(let b=0;b<999;b++){const sample=shuffle(others,random).slice(0,k),wl=sample.reduce((a,c)=>a+c,0)/k;sim.push(z[i]*wl/m2)}const p=(1+sim.filter(x=>Math.abs(x)>=Math.abs(obs)).length)/1000;let type='NS';if(p<.05)type=z[i]>=0?(l[i]>=0?'HH':'HL'):(l[i]>=0?'LH':'LL');return{city,I:obs,p,z:z[i],lag:l[i],type}})}
const levelYears=[1990,2000,2010,2020],periods=['1990-2000','2000-2010','2010-2020'];
const level=[];
for(const year of levelYears){const rs=names.map(n=>metrics.rows.find(r=>norm(r.city)===n&&r.year===year)),global={};for(const m of['oe','pte','se'])global[m]=globalTest(rs.map(r=>r[m]),year+({oe:11,pte:29,se:47}[m]));level.push({year,global,oe_lisa:lisa(rs.map(r=>r.oe),year+101)})}
const change=[];
for(let pi=0;pi<periods.length;pi++){const period=periods[pi],rs=names.map(n=>metrics.gml_rows.find(r=>norm(r.city)===n&&r.period===period)),global={};for(const m of['gml','tc','pec','sec'])global[m]=globalTest(rs.map(r=>Math.log(r[m])),6000+pi*100+({gml:11,tc:29,pec:47,sec:65}[m]));change.push({period,global,gml_lisa:lisa(rs.map(r=>Math.log(r.gml)),7000+pi)})}
const out={weights:{type:'Queen contiguity + island nearest-neighbor correction',row_standardized:true,permutations:999,corrections:['舟山–宁波 added; 舟山–上海 retained'],edges:[...adj.values()].reduce((s,x)=>s+x.size,0)/2,degrees:Object.fromEntries(names.map(n=>[n,adj.get(n).size]))},change_transform:'natural logarithm',level,change};
await fs.writeFile(`${root}/results/derived/spatial_results_figure6.json`,JSON.stringify(out,null,2));
console.log(JSON.stringify({level:level.map(r=>({year:r.year,...Object.fromEntries(Object.entries(r.global).map(([k,v])=>[k,{I:v.I,p:v.p}]))})),change:change.map(r=>({period:r.period,...Object.fromEntries(Object.entries(r.global).map(([k,v])=>[k,{I:v.I,p:v.p}]))}))},null,2));
