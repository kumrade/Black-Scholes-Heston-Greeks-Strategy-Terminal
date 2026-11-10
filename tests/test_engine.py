"""Deterministic mathematical fixtures, not application demo/market data."""
import csv
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy.integrate import quad

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('research_main',ROOT/'main.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
P=dict(kappa=2.0,theta=.04,xi=.3,rho=-.5,v0=.04)


def fixture_quotes(T=.5):
    # Model-derived test inputs only. They are never bundled as market quotes.
    return [dict(strike=K,ce=m.bs_price(100,K,T,.22,.03,.01,True),
                 pe=m.bs_price(100,K,T,.22,.03,.01,False))
            for K in (80.,90.,100.,110.,120.)]


class EngineTests(unittest.TestCase):
    def test_bsm_parity_and_inverse(self):
        for K in (80,100,120):
            c=m.bs_price(100,K,.5,.22,.03,.01,True)
            p=m.bs_price(100,K,.5,.22,.03,.01,False)
            self.assertAlmostEqual(c-p,100*math.exp(-.01*.5)-K*math.exp(-.03*.5),places=10)
            self.assertAlmostEqual(m.iv_solve(c,100,K,.5,.03,.01,True),.22,places=6)
            self.assertAlmostEqual(m.iv_solve(p,100,K,.5,.03,.01,False),.22,places=6)

    def test_discounted_iv_bounds(self):
        self.assertTrue(math.isnan(m.iv_solve(101,100,100,.5,.03,.01,True)))
        self.assertTrue(math.isnan(m.iv_solve(-1,100,100,.5,.03,.01,True)))
        low,_=m.price_bounds(100,90,.5,.03,.01,True)
        self.assertTrue(math.isnan(m.iv_solve(low-1,100,90,.5,.03,.01,True)))

    def test_zero_vol_and_expiry(self):
        self.assertEqual(m.bs_price(100,90,0,.2,.03,.01,True),10)
        self.assertAlmostEqual(m.bs_price(100,90,.5,0,.03,.01,True),
                               100*math.exp(-.005)-90*math.exp(-.015))

    def test_fft_against_independent_inversion(self):
        S,T,r,q=100,.5,.03,.01
        phi=lambda u:m.heston_cf(u,S,T,r,q,*P.values())
        for K in (90,100,110):
            def kernel(u,which):
                if which==1:
                    return float(np.real(np.exp(-1j*u*np.log(K))*phi(u-1j)/(1j*u*phi(-1j))))
                return float(np.real(np.exp(-1j*u*np.log(K))*phi(u)/(1j*u)))
            p1=.5+quad(lambda u:kernel(u,1),1e-8,200,epsabs=1e-7)[0]/math.pi
            p2=.5+quad(lambda u:kernel(u,2),1e-8,200,epsabs=1e-7)[0]/math.pi
            reference=S*math.exp(-q*T)*p1-K*math.exp(-r*T)*p2
            ce,pe=m.heston_fft(S,[K],T,r,q,*P.values(),N=16384,eta=.1)
            self.assertLess(abs(ce[0]-reference),.002)
            self.assertAlmostEqual(ce[0]-pe[0],S*math.exp(-q*T)-K*math.exp(-r*T),places=10)

    def test_heston_initial_vol_chain_rule(self):
        g=m.heston_greeks(100,100,.5,.03,.01,P,hv=1e-5)
        sig=math.sqrt(P['v0']);bump=1e-5
        def value(sig0):
            pp=dict(P,v0=sig0**2)
            return m.heston_fft(100,[100],.5,.03,.01,*pp.values())[0][0]
        direct=(value(sig+bump)-value(sig-bump))/(2*bump)/100
        self.assertAlmostEqual(g['initial_vol_vega_01'],direct,places=5)

    def test_all_strategy_mappings_and_missing_name(self):
        self.assertEqual(len(m.STRATEGIES),31)
        for name in m.STRATEGIES:
            n=1 if name in ('Long Call','Short Call','Long Put','Short Put',
                'Covered Call','Protective Put','Cash-Secured Put','Long Straddle',
                'Short Straddle','Calendar Spread') else (3 if 'Butterfly' in name else (4 if 'Condor' in name else 2))
            legs,_,_=m.strategy_spec(name,[90+10*i for i in range(n)])
            self.assertTrue(legs)
        with self.assertRaises(ValueError):m.strategy_spec('typo',[100])
        with self.assertRaises(ValueError):m.strategy_spec('Bear Call Spread',[110,100])

    def test_bear_call_payoff_is_capped(self):
        book=m.build_strategy('Bear Call Spread',[100,110],fixture_quotes(),100,.5,.03,.01)
        self.assertEqual([(x['opt_type'],x['qty'])for x in book['legs']],[('CE',-1),('CE',1)])
        credit=-book['net_greeks']['price']
        values=m.strategy_pnl(book['legs'],[50,150,200],dT=182.5)
        np.testing.assert_allclose(values,[credit,credit-10,credit-10],atol=1e-10)

    def test_calendar_near_expiry_retains_far_time_value(self):
        book=m.build_strategy('Calendar Spread',[100],fixture_quotes(.5),100,.5,.03,.01,fixture_quotes(1.),1.)
        value=m.strategy_pnl(book['legs'],[100],dT=182.5)[0]
        expected=m.bs_price(100,100,.5,.22,.03,.01,True)-book['net_greeks']['price']
        self.assertAlmostEqual(value,expected,places=6)
        with self.assertRaises(ValueError):
            m.build_strategy('Calendar Spread',[100],fixture_quotes(),100,.5,.03,.01)

    def test_no_network_or_gui_imports(self):
        import ast
        tree=ast.parse((ROOT/'main.py').read_text())
        roots=set()
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):roots.update(a.name.split('.')[0] for a in node.names)
            elif isinstance(node,ast.ImportFrom) and node.module:roots.add(node.module.split('.')[0])
        self.assertFalse(roots.intersection({'SmartApi','pyotp','requests','urllib','http','socket','tkinter','websocket'}))

    def test_bad_csv_and_timezone(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'bad.csv';path.write_text('strike,ce,pe\n100,,5\n')
            with self.assertRaises(ValueError):m.read_quotes(path,100,.5,.03,.01)
        with self.assertRaises(ValueError):m.parse_instant('2026-11-11T12:00:00')

    def test_cli_end_to_end_with_explicit_parameters(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);quotes=base/'quotes.csv';params=base/'params.json';out=base/'results'
            with quotes.open('w',newline='')as handle:
                writer=csv.DictWriter(handle,fieldnames=['strike','ce','pe']);writer.writeheader();writer.writerows(fixture_quotes())
            params.write_text(json.dumps(P))
            command=[sys.executable,str(ROOT/'main.py'),'--quotes',str(quotes),
                '--spot','100','--rate','.03','--dividend','.01',
                '--as-of','2026-01-01T00:00:00+00:00','--expiry','2026-07-02T12:00:00+00:00',
                '--params',str(params),'--strategy','Bear Call Spread','--strikes','100','110',
                '--spot-change','2','--iv-change-pp','1','--days','1','--output',str(out)]
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            payload=json.loads((out/'analysis.json').read_text())
            self.assertEqual(len(payload['chain']),5)
            self.assertEqual(payload['strategy']['legs'][0]['qty'],-1)
            self.assertTrue((out/'strategy_pnl.csv').exists())
            self.assertTrue(all(row['heston_iv'] is not None for row in payload['chain']))
            second=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(second.returncode,0)


if __name__=='__main__':unittest.main()
