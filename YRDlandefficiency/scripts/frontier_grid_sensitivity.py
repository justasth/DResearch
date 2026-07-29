import json
import math
import sys
from pathlib import Path

import numpy as np
import shapefile
from pyproj import Transformer
from scipy.stats import linregress
from shapely.geometry import box, shape
from shapely.ops import transform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from compute_metrics import load_data, sbm_undesirable, directional_distance, YEARS


def grid_exposure():
    reader = shapefile.Reader(str(ROOT / "data" / "boundaries" / "长三角市级边界.shp"), encoding="utf-8")
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
    cities = {}
    for sr in reader.iterShapeRecords():
        name = sr.record[0]
        geom = shape(sr.shape.__geo_interface__)
        cities[name] = geom

    xmin = math.floor(min(g.bounds[0] for g in cities.values()) * 10) / 10
    xmax = math.ceil(max(g.bounds[2] for g in cities.values()) * 10) / 10
    ymin = math.floor(min(g.bounds[1] for g in cities.values()) * 10) / 10
    ymax = math.ceil(max(g.bounds[3] for g in cities.values()) * 10) / 10
    cells = []
    x = xmin
    while x < xmax - 1e-9:
        y = ymin
        while y < ymax - 1e-9:
            cell = box(round(x, 10), round(y, 10), round(x + .1, 10), round(y + .1, 10))
            hits = [name for name, geom in cities.items() if geom.intersects(cell) and not geom.intersection(cell).is_empty]
            if hits:
                cells.append((cell, hits))
            y += .1
        x += .1

    out = []
    for name, geom in cities.items():
        total_area = transform(transformer.transform, geom).area
        intersecting = 0
        shared = 0
        shared_area = 0.0
        partial = 0
        partial_area = 0.0
        for cell, hits in cells:
            if name not in hits:
                continue
            inter = geom.intersection(cell)
            if inter.is_empty:
                continue
            intersecting += 1
            ia = transform(transformer.transform, inter).area
            if len(hits) > 1:
                shared += 1
                shared_area += ia
            # A partial cell is one not fully covered by this city's polygon.
            if not geom.covers(cell):
                partial += 1
                partial_area += ia
        out.append({
            "city": name,
            "intersecting_cells": intersecting,
            "shared_cells": shared,
            "partial_cells": partial,
            "shared_city_area_share": shared_area / total_area,
            "partial_city_area_share": partial_area / total_area,
        })
    return sorted(out, key=lambda z: z["city"])


def calculate(rows, dynamic=True):
    X = np.array([[r["land"]] for r in rows], dtype=float)
    YG = np.array([[r["population"], r["real_gdp"]] for r in rows], dtype=float)
    YB = np.array([[r["co2"]] for r in rows], dtype=float)
    out_rows = [dict(r) for r in rows]
    for i, r in enumerate(out_rows):
        r["oe"] = sbm_undesirable(X[i], YG[i], YB[i], X, YG, YB, False)
    annual = []
    for year in YEARS:
        vals = np.array([r["oe"] for r in out_rows if r["year"] == year])
        annual.append({"year": year, "mean": float(vals.mean()), "cv": float(vals.std(ddof=1) / vals.mean())})
    beta = []
    for y0, y1 in zip(YEARS[:-1], YEARS[1:]):
        a = sorted([r for r in out_rows if r["year"] == y0], key=lambda z: z["city"])
        b = sorted([r for r in out_rows if r["year"] == y1], key=lambda z: z["city"])
        e0 = np.array([r["oe"] for r in a]); e1 = np.array([r["oe"] for r in b])
        lr = linregress(np.log(e0), np.log(e1/e0)/(y1-y0))
        beta.append({"period": f"{y0}-{y1}", "beta": float(lr.slope), "p": float(lr.pvalue)})
    result = {"n_cities": len({r['city'] for r in rows}), "annual": annual, "beta": beta}
    if not dynamic:
        return result

    by_year = {y: [i for i, r in enumerate(out_rows) if r["year"] == y] for y in YEARS}
    for i, r in enumerate(out_rows):
        own = by_year[r["year"]]
        r["dg"] = directional_distance(X[i], YG[i], YB[i], X, YG, YB, False)
        r["dc"] = directional_distance(X[i], YG[i], YB[i], X[own], YG[own], YB[own], False)
        r["dv"] = directional_distance(X[i], YG[i], YB[i], X[own], YG[own], YB[own], True)
    periods = []
    gm = lambda vals: float(np.exp(np.mean(np.log(np.asarray(vals)))))
    for y0, y1 in zip(YEARS[:-1], YEARS[1:]):
        a = sorted([r for r in out_rows if r["year"] == y0], key=lambda z: z["city"])
        b = sorted([r for r in out_rows if r["year"] == y1], key=lambda z: z["city"])
        vals=[]
        for ra, rb in zip(a,b):
            gml=(1+ra['dg'])/(1+rb['dg'])
            ec=(1+ra['dc'])/(1+rb['dc'])
            pec=(1+ra['dv'])/(1+rb['dv'])
            vals.append((gml,gml/ec,ec,pec,ec/pec))
        periods.append({"period": f"{y0}-{y1}", "gml": gm([v[0] for v in vals]),
                        "tc": gm([v[1] for v in vals]), "ec": gm([v[2] for v in vals]),
                        "pec": gm([v[3] for v in vals]), "sec": gm([v[4] for v in vals])})
    result["gml"] = periods
    return result


def main():
    rows = load_data()
    exposure = grid_exposure()
    shares = np.array([r["shared_city_area_share"] for r in exposure])
    q75 = float(np.quantile(shares, .75))
    high_exposure = sorted(r["city"] for r in exposure if r["shared_city_area_share"] >= q75)
    frontier = ["丽水市", "舟山市", "南通市", "温州市", "上海市", "扬州市", "蚌埠市"]
    scenarios = {
        "main": [],
        "exclude_shanghai": ["上海市"],
        "exclude_all_observed_frontier_cities": frontier,
        "exclude_top_quartile_grid_exposure": high_exposure,
    }
    scenario_results = {}
    for key, excluded in scenarios.items():
        print("running", key, len(excluded), flush=True)
        scenario_results[key] = {"excluded": excluded,
                                 "results": calculate([r for r in rows if r["city"] not in excluded], dynamic=True)}
    loo = []
    for city in sorted({r["city"] for r in rows}):
        print("loo", city, flush=True)
        res = calculate([r for r in rows if r["city"] != city], dynamic=False)
        loo.append({"excluded": city, **res})
    payload = {"grid_exposure": exposure, "grid_exposure_q75": q75,
               "high_exposure_cities": high_exposure, "scenarios": scenario_results,
               "leave_one_city_out": loo}
    (ROOT / "results" / "derived" / "frontier_grid_sensitivity.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"q75":q75,"high_exposure":high_exposure,
                      "scenarios":scenario_results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
