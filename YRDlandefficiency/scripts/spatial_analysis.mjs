import fs from "node:fs/promises";

import { fileURLToPath } from "node:url";
const root=fileURLToPath(new URL("..",import.meta.url));
const codes=[310000,320000,330000,340000], features=[];
for(const code of codes){const suffix=code===310000?"":"_full";const j=JSON.parse(await fs.readFile(`${root}/data/boundaries/${code}${suffix}.json`,"utf8"));for(const f of j.features)features.push({code,...f});}
const metrics=JSON.parse(await fs.readFile(`${root}/results/derived/computed_metrics.json`,"utf8"));
const norm=n=>n.replace(/市$/u,"");
function pts(g){return (g.type==="Polygon"?g.coordinates:g.coordinates.flat()).flat();}
const vertex=new Map();
for(const f of features){const n=f.code===310000?"上海":norm(f.properties.name);if(!vertex.has(n))vertex.set(n,new Set());for(const [x,y] of pts(f.geometry))vertex.get(n).add(`${x.toFixed(4)},${y.toFixed(4)}`)}
const names=[...new Set(metrics.rows.map(r=>norm(r.city)))].sort((a,b)=>a.localeCompare(b,"zh-CN"));
const adj=new Map(names.map(n=>[n,new Set()]));
for(let i=0;i<names.length;i++)for(let j=i+1;j<names.length;j++){
  const a=vertex.get(names[i]),b=vertex.get(names[j]);let hit=false;
  for(const q of a){if(b.has(q)){hit=true;break}}
  if(hit){adj.get(names[i]).add(names[j]);adj.get(names[j]).add(names[i]);}
}
// Island correction: retain the Queen-derived Shanghai–Zhoushan maritime
// adjacency and add Zhoushan–Ningbo as the nearest mainland connection.
adj.get("舟山").add("宁波");adj.get("宁波").add("舟山");
if(names.some(n=>adj.get(n).size===0))throw new Error(`Isolated cities: ${names.filter(n=>adj.get(n).size===0)}`);
const idx=new Map(names.map((n,i)=>[n,i]));
const W=names.map(n=>{const row=Array(names.length).fill(0),d=adj.get(n).size;for(const q of adj.get(n))row[idx.get(q)]=1/d;return row;});

function mulLag(v){return W.map(row=>row.reduce((s,w,j)=>s+w*v[j],0));}
function moran(vals){const mean=vals.reduce((a,b)=>a+b,0)/vals.length,z=vals.map(v=>v-mean),lag=mulLag(z);return z.reduce((s,v,i)=>s+v*lag[i],0)/z.reduce((s,v)=>s+v*v,0);}
function rng(seed){return()=>{seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function shuffle(a,random){const x=[...a];for(let i=x.length-1;i>0;i--){const j=Math.floor(random()*(i+1));[x[i],x[j]]=[x[j],x[i]]}return x;}
function globalTest(vals,seed){const obs=moran(vals),random=rng(seed),perms=[];for(let k=0;k<999;k++)perms.push(moran(shuffle(vals,random)));const p=(1+perms.filter(x=>Math.abs(x)>=Math.abs(obs)).length)/1000;return{I:obs,p,perm_mean:perms.reduce((a,b)=>a+b,0)/perms.length};}
function lisa(vals,seed){const n=vals.length,mean=vals.reduce((a,b)=>a+b,0)/n,z=vals.map(v=>v-mean),m2=z.reduce((s,v)=>s+v*v,0)/n,lag=mulLag(z),random=rng(seed),out=[];
  for(let i=0;i<n;i++){
    const obs=z[i]*lag[i]/m2,k=adj.get(names[i]).size,others=z.filter((_,j)=>j!==i),sim=[];
    for(let b=0;b<999;b++){const sample=shuffle(others,random).slice(0,k),l=sample.reduce((a,c)=>a+c,0)/k;sim.push(z[i]*l/m2)}
    const p=(1+sim.filter(x=>Math.abs(x)>=Math.abs(obs)).length)/1000;
    let type="NS";if(p<.05)type=z[i]>=0?(lag[i]>=0?"HH":"HL"):(lag[i]>=0?"LH":"LL");
    out.push({city:names[i],I:obs,p,z:z[i],lag:lag[i],type,neighbors:[...adj.get(names[i])].sort()});
  }return out;
}
const result={weights:{type:"Queen contiguity + island nearest-neighbor correction",rounding:"4 decimal degrees",row_standardized:true,corrections:["舟山–宁波 added; 舟山–上海 retained"],edges:[...adj.values()].reduce((s,x)=>s+x.size,0)/2,degrees:Object.fromEntries(names.map(n=>[n,adj.get(n).size]))},years:[]};
for(const year of [1990,2000,2010,2020]){
  const rows=names.map(n=>metrics.rows.find(r=>norm(r.city)===n&&r.year===year));
  const oe=rows.map(r=>r.oe),pte=rows.map(r=>r.pte);
  result.years.push({year,oe_global:globalTest(oe,year+11),pte_global:globalTest(pte,year+29),oe_lisa:lisa(oe,year+101),pte_lisa:lisa(pte,year+211)});
}
await fs.writeFile(`${root}/results/derived/spatial_results.json`,JSON.stringify(result,null,2));
console.log(JSON.stringify({weights:result.weights,global:result.years.map(y=>({year:y.year,OE:y.oe_global,PTE:y.pte_global,OE_types:Object.fromEntries(["HH","LL","HL","LH","NS"].map(t=>[t,y.oe_lisa.filter(x=>x.type===t).length]))}))},null,2));
