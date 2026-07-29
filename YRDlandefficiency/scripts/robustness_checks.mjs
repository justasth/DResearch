import fs from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const metrics = JSON.parse(await fs.readFile(`${root}/results/derived/computed_metrics.json`, 'utf8'));
const codes = [310000, 320000, 330000, 340000];
const features = [];
for (const code of codes) {
  const suffix = code === 310000 ? '' : '_full';
  const j = JSON.parse(await fs.readFile(`${root}/data/boundaries/${code}${suffix}.json`, 'utf8'));
  for (const f of j.features) features.push({code, ...f});
}
const norm = n => String(n).replace(/市$/u, '');
const names = [...new Set(metrics.rows.map(r => norm(r.city)))].sort((a,b)=>a.localeCompare(b,'zh-CN'));
const idx = new Map(names.map((n,i)=>[n,i]));

function rings(g) {
  if (g.type === 'Polygon') return g.coordinates;
  return g.coordinates.flat();
}
function allPoints(g) { return rings(g).flat(); }
function polygonCentroid(g) {
  let sx=0, sy=0, sw=0;
  for (const ring of rings(g)) {
    let a=0,cx=0,cy=0;
    for(let i=0;i<ring.length-1;i++){
      const [x0,y0]=ring[i], [x1,y1]=ring[i+1];
      const q=x0*y1-x1*y0; a+=q; cx+=(x0+x1)*q; cy+=(y0+y1)*q;
    }
    if(Math.abs(a)>1e-12){ const area=a/2; sx += area*cx/(6*area); sy += area*cy/(6*area); sw += area; }
  }
  if(Math.abs(sw)>1e-12) return [sx/sw,sy/sw];
  const p=allPoints(g); return [p.reduce((s,z)=>s+z[0],0)/p.length,p.reduce((s,z)=>s+z[1],0)/p.length];
}

const geom = new Map();
for(const f of features) geom.set(f.code===310000?'上海':norm(f.properties.name), f.geometry);
const vertices = new Map(names.map(n=>[n,new Set(allPoints(geom.get(n)).map(([x,y])=>`${x.toFixed(4)},${y.toFixed(4)}`))]));
function queen(addNbo=true){
  const adj=names.map(()=>new Set());
  for(let i=0;i<names.length;i++)for(let j=i+1;j<names.length;j++){
    let hit=false; for(const q of vertices.get(names[i])) if(vertices.get(names[j]).has(q)){hit=true;break;}
    if(hit){adj[i].add(j);adj[j].add(i);}
  }
  if(addNbo){const a=idx.get('舟山'),b=idx.get('宁波');adj[a].add(b);adj[b].add(a);}
  return adj.map(s=>{const row=Array(names.length).fill(0);for(const j of s)row[j]=1/s.size;return row;});
}
const cent = names.map(n=>polygonCentroid(geom.get(n)));
function hav([x1,y1],[x2,y2]){const r=Math.PI/180, a=Math.sin((y2-y1)*r/2)**2+Math.cos(y1*r)*Math.cos(y2*r)*Math.sin((x2-x1)*r/2)**2;return 2*6371*Math.asin(Math.sqrt(a));}
function knn(k){return cent.map((c,i)=>{const js=cent.map((d,j)=>[j,hav(c,d)]).filter(z=>z[0]!==i).sort((a,b)=>a[1]-b[1]).slice(0,k);const row=Array(names.length).fill(0);for(const [j] of js)row[j]=1/k;return row;});}
function moran(v,W){const n=v.length,m=v.reduce((a,b)=>a+b,0)/n,z=v.map(x=>x-m),lag=W.map(r=>r.reduce((s,w,j)=>s+w*z[j],0));return z.reduce((s,x,i)=>s+x*lag[i],0)/z.reduce((s,x)=>s+x*x,0);}
function rng(seed=20260722){return()=>{seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
function permP(v,W,B=999){const obs=moran(v,W), rr=rng(), sims=[];for(let b=0;b<B;b++){const p=[...v];for(let i=p.length-1;i>0;i--){const j=Math.floor(rr()*(i+1));[p[i],p[j]]=[p[j],p[i]];}sims.push(moran(p,W));}const ei=-1/(v.length-1);return {I:obs,p:(1+sims.filter(x=>Math.abs(x-ei)>=Math.abs(obs-ei)).length)/(B+1)};}
function inv2([[a,b],[c,d]]){const q=a*d-b*c;return [[d/q,-b/q],[-c/q,a/q]];}
function mm(A,B){return A.map(r=>B[0].map((_,j)=>r.reduce((s,x,k)=>s+x*B[k][j],0)));}
function olsRobust(x,y){const n=x.length,k=2,Sx=x.reduce((s,z)=>s+z,0),Sxx=x.reduce((s,z)=>s+z*z,0),Sy=y.reduce((s,z)=>s+z,0),Sxy=x.reduce((s,z,i)=>s+z*y[i],0);const inv=inv2([[n,Sx],[Sx,Sxx]]), bh=[inv[0][0]*Sy+inv[0][1]*Sxy,inv[1][0]*Sy+inv[1][1]*Sxy],e=y.map((z,i)=>z-bh[0]-bh[1]*x[i]);let meat=[[0,0],[0,0]];for(let i=0;i<n;i++){const z=[1,x[i]],q=e[i]*e[i]*n/(n-k);for(let a=0;a<2;a++)for(let b=0;b<2;b++)meat[a][b]+=q*z[a]*z[b];}const v=mm(mm(inv,meat),inv);return {alpha:bh[0],beta:bh[1],hc1_se:Math.sqrt(v[1][1]),residuals:e};}

const Wq=queen(true), Ws={queen:Wq,knn4:knn(4),knn6:knn(6)};
const spatial={};
for(const [label,W] of Object.entries(Ws)){spatial[label]={};for(const year of [1990,2000,2010,2020]){const v=names.map(n=>metrics.rows.find(r=>norm(r.city)===n&&r.year===year).oe);spatial[label][year]=permP(v,W);}}
const beta=[];
for(const [y0,y1] of [[1990,2000],[2000,2010],[2010,2020]]){
  const e0=names.map(n=>metrics.rows.find(r=>norm(r.city)===n&&r.year===y0).oe), e1=names.map(n=>metrics.rows.find(r=>norm(r.city)===n&&r.year===y1).oe);
  const x=e0.map(Math.log),y=e0.map((z,i)=>Math.log(e1[i]/z)/(y1-y0));const fit=olsRobust(x,y), rm=permP(fit.residuals,Wq);
  beta.push({period:`${y0}-${y1}`,beta:fit.beta,hc1_se:fit.hc1_se,hc1_t:fit.beta/fit.hc1_se,residual_moran:rm});
}
const out={spatial,beta};
await fs.writeFile(`${root}/results/derived/robustness_results.json`,JSON.stringify(out,null,2));
console.log(JSON.stringify(out,null,2));
