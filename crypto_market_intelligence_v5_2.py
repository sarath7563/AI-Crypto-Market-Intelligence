import os, sqlite3, time, hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

load_dotenv()
st.set_page_config(page_title="AI Crypto Research Terminal V5.2", page_icon="🧠", layout="wide")

APP_TITLE="AI Crypto Research Terminal V5.2"
DB_PATH=Path("crypto_market.db")
CACHE_DIR=Path(".crypto_cache_v5"); CACHE_DIR.mkdir(exist_ok=True)
BINANCE="https://api.binance.com/api/v3"
COINGECKO="https://api.coingecko.com/api/v3"
COINPAPRIKA="https://api.coinpaprika.com/v1"
CG_KEY=os.getenv("COINGECKO_API_KEY","").strip()
CP_KEY=os.getenv("COINPAPRIKA_API_KEY","").strip()
CG_HEADERS={"accept":"application/json"};
if CG_KEY: CG_HEADERS["x-cg-demo-api-key"]=CG_KEY

# ---------------- HTTP + cache ----------------
def cache_path(key): return CACHE_DIR/(hashlib.sha256(key.encode()).hexdigest()+".pkl")

def get_json(base,path,params=None,ttl=300,retries=2):
    params=params or {}
    key=base+path+"?"+"&".join(f"{k}={params[k]}" for k in sorted(params))
    fp=cache_path(key)
    if fp.exists() and time.time()-fp.stat().st_mtime<ttl:
        return pd.read_pickle(fp)
    last=None
    for a in range(retries+1):
        try:
            headers=CG_HEADERS if "coingecko" in base else {"accept":"application/json"}
            r=requests.get(base+path,params=params,headers=headers,timeout=25)
            if r.status_code==429:
                last=RuntimeError("HTTP 429 rate limit")
                if a<retries: time.sleep(3*(a+1)); continue
                raise last
            r.raise_for_status(); data=r.json(); pd.to_pickle(data,fp); return data
        except Exception as e:
            last=e
            if a<retries: time.sleep(2*(a+1))
    raise RuntimeError(str(last))

# ---------------- catalogue ----------------
def db():
    con=sqlite3.connect(DB_PATH)
    con.execute("CREATE TABLE IF NOT EXISTS coin_catalog (id TEXT PRIMARY KEY, symbol TEXT, name TEXT, last_synced TEXT)")
    con.commit(); return con

def sync_catalog():
    data=get_json(COINGECKO,"/coins/list",{"include_platform":"false"},86400,1)
    now=datetime.now(timezone.utc).isoformat(); con=db()
    con.executemany("INSERT OR REPLACE INTO coin_catalog VALUES (?,?,?,?)",[(x.get('id',''),x.get('symbol',''),x.get('name',''),now) for x in data if x.get('id')])
    con.commit(); n=con.execute("SELECT COUNT(*) FROM coin_catalog").fetchone()[0]; con.close(); return n

def search_catalog(q,limit=100):
    con=db(); q=(q or '').strip().lower(); like=f"%{q}%"
    if q:
        rows=con.execute("SELECT id,symbol,name FROM coin_catalog WHERE lower(name) LIKE ? OR lower(symbol) LIKE ? OR lower(id) LIKE ? ORDER BY CASE WHEN lower(symbol)=? THEN 0 WHEN lower(id)=? THEN 1 WHEN lower(name)=? THEN 2 ELSE 3 END,name LIMIT ?",(like,like,like,q,q,q,limit)).fetchall()
    else: rows=con.execute("SELECT id,symbol,name FROM coin_catalog ORDER BY name LIMIT ?",(limit,)).fetchall()
    con.close(); return pd.DataFrame(rows,columns=['id','symbol','name'])

# ---------------- market data ----------------
def binance_klines(symbol, interval, limit=1000):
    sym=str(symbol).upper().replace('-','').replace(' ','')+'USDT'
    try:
        rows=get_json(BINANCE,"/klines",{"symbol":sym,"interval":interval,"limit":limit},300,2)
        if not rows: return pd.DataFrame()
        cols=['open_time','open','high','low','close','volume','close_time','quote_volume','trades','taker_buy_base','taker_buy_quote','ignore']
        d=pd.DataFrame(rows,columns=cols)
        for c in ['open','high','low','close','volume','quote_volume']: d[c]=pd.to_numeric(d[c],errors='coerce')
        d['date']=pd.to_datetime(d.open_time,unit='ms',utc=True)
        return d[['date','open','high','low','close','volume','quote_volume','trades']].dropna().reset_index(drop=True)
    except Exception: return pd.DataFrame()

def cp_search(name, symbol):
    q=str(symbol or name or "").strip()
    if not q: return None
    try:
        data=get_json(COINPAPRIKA,"/search/",{"q":q,"c":"currencies","limit":10},86400,1)
        items=data.get("currencies",[]) if isinstance(data,dict) else []
        target_sym=str(symbol or "").lower()
        target_name=str(name or "").lower()
        exact=[x for x in items if str(x.get("symbol","" )).lower()==target_sym]
        if exact: return exact[0].get("id")
        exact=[x for x in items if str(x.get("name","" )).lower()==target_name]
        if exact: return exact[0].get("id")
        return items[0].get("id") if items else None
    except Exception:
        return None

def cp_daily(name, symbol, days=365):
    cp_id=cp_search(name, symbol)
    if not cp_id: raise ValueError("CoinGecko rejected the request and CoinPaprika could not map this coin.")
    end=pd.Timestamp.now(tz="UTC").floor("D")
    start=end-pd.Timedelta(days=min(int(days),365))
    params={"start":start.strftime("%Y-%m-%d"),"end":end.strftime("%Y-%m-%d"),"interval":"1d"}
    data=get_json(COINPAPRIKA,f"/tickers/{cp_id}/historical",params,21600,1)
    rows=data if isinstance(data,list) else []
    if not rows: raise ValueError("No historical daily data returned by the fallback provider.")
    p=pd.DataFrame(rows)
    p["date"]=pd.to_datetime(p.get("timestamp",p.get("date")),utc=True,errors="coerce")
    if "price" not in p.columns: raise ValueError("Fallback provider returned no price series.")
    p["close"]=pd.to_numeric(p["price"],errors="coerce")
    p["volume"]=pd.to_numeric(p.get("volume_24h"),errors="coerce")
    p=p.dropna(subset=["date","close"]).sort_values("date").drop_duplicates("date")
    p["open"]=p["close"].shift(1).fillna(p["close"])
    p["high"]=p[["open","close"]].max(axis=1)
    p["low"]=p[["open","close"]].min(axis=1)
    return p[["date","open","high","low","close","volume"]].reset_index(drop=True)

def cg_daily(coin_id,days=730, name="", symbol=""):
    try:
        data=get_json(COINGECKO,f"/coins/{coin_id}/market_chart",{'vs_currency':'usd','days':str(days),'interval':'daily'},900,1)
        p=pd.DataFrame(data.get('prices',[]),columns=['ts','close']); v=pd.DataFrame(data.get('total_volumes',[]),columns=['ts','volume'])
        if p.empty: raise ValueError('No historical price data returned.')
        p['date']=pd.to_datetime(p.ts,unit='ms',utc=True).dt.floor('D'); p=p.groupby('date',as_index=False).last()
        if not v.empty:
            v['date']=pd.to_datetime(v.ts,unit='ms',utc=True).dt.floor('D'); v=v.groupby('date',as_index=False).last()[['date','volume']]
            p=p.merge(v,on='date',how='left')
        p['open']=p.close.shift(1).fillna(p.close); p['high']=p[['open','close']].max(axis=1); p['low']=p[['open','close']].min(axis=1)
        return p[['date','open','high','low','close','volume']].dropna(subset=['close']).reset_index(drop=True), 'CoinGecko daily'
    except Exception as cg_error:
        try:
            return cp_daily(name,symbol,min(days,365)), 'CoinPaprika daily fallback'
        except Exception as cp_error:
            raise RuntimeError(f"CoinGecko authentication/data request failed ({cg_error}). Fallback also failed ({cp_error}). Set COINGECKO_API_KEY in .env or use a coin available on Binance.")

def load_base(sel,days=730):
    b=binance_klines(sel.symbol,'1d',min(1000,max(365,days)))
    if len(b)>=250: return b,'Binance Spot'
    d,source=cg_daily(sel.id,days,str(sel['name']),str(sel['symbol']))
    return d,source

def timeframe_data(sel,tf):
    # Prefer native Binance candles for intraday. For 2D/1W resample daily data.
    if tf in {'1h','4h'}:
        b=binance_klines(sel.symbol,tf,1000)
        if len(b)>=220: return b,'Binance Spot'
        base,_=load_base(sel,365)
        return base,'CoinGecko daily fallback (intraday timeframe unavailable)'
    base,source=load_base(sel,365)
    return resample_ohlc(base,'2D' if tf=='2D' else '1W'),source

def resample_ohlc(d,rule):
    x=d.set_index('date').sort_index()
    agg={'open':'first','high':'max','low':'min','close':'last','volume':'sum'}
    return x.resample(rule).agg(agg).dropna(subset=['close']).reset_index()

# ---------------- indicators / structure ----------------
def add_indicators(d):
    x=d.copy(); c=x.close; h=x.high; l=x.low; v=x.volume
    for n in [20,50,200]: x[f'ema{n}']=c.ewm(span=n,adjust=False).mean(); x[f'sma{n}']=c.rolling(n).mean()
    delta=c.diff(); gain=delta.clip(lower=0).rolling(14).mean(); loss=(-delta.clip(upper=0)).rolling(14).mean(); rs=gain/loss.replace(0,np.nan); x['rsi']=100-100/(1+rs)
    e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean(); x['macd']=e12-e26; x['macd_signal']=x.macd.ewm(span=9,adjust=False).mean(); x['macd_hist']=x.macd-x.macd_signal
    mid=c.rolling(20).mean(); sd=c.rolling(20).std(); x['bb_mid']=mid; x['bb_up']=mid+2*sd; x['bb_dn']=mid-2*sd
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1); x['atr']=tr.rolling(14).mean(); x['atr_pct']=100*x.atr/c
    low14=l.rolling(14).min(); high14=h.rolling(14).max(); x['stoch_k']=100*(c-low14)/(high14-low14).replace(0,np.nan); x['stoch_d']=x.stoch_k.rolling(3).mean()
    up=h.diff(); down=-l.diff(); plus=np.where((up>down)&(up>0),up,0.0); minus=np.where((down>up)&(down>0),down,0.0); atr=x.atr.replace(0,np.nan); dip=100*pd.Series(plus,index=x.index).rolling(14).mean()/atr; dim=100*pd.Series(minus,index=x.index).rolling(14).mean()/atr; dx=100*(dip-dim).abs()/(dip+dim).replace(0,np.nan); x['adx']=dx.rolling(14).mean()
    x['vol_ma']=v.rolling(20).mean(); x['vol_ratio']=v/x.vol_ma.replace(0,np.nan); x['obv']=(np.sign(c.diff()).fillna(0)*v.fillna(0)).cumsum(); x['ret1']=c.pct_change()*100; x['ret7']=c.pct_change(7)*100; x['ret30']=c.pct_change(30)*100
    x['future5']=c.shift(-5)/c.sub(0).replace(0,np.nan)-1; x['future5']*=100; x['target']=(x.future5>0).astype(int)
    return x

def pivots(d,left=3,right=3):
    x=d.copy(); hi=x.high.values; lo=x.low.values; sh=np.full(len(x),False); sl=np.full(len(x),False)
    for i in range(left,len(x)-right):
        sh[i]=hi[i]>=max(hi[i-left:i+right+1]); sl[i]=lo[i]<=min(lo[i-left:i+right+1])
    x['swing_high']=sh; x['swing_low']=sl
    return x

def structure(d):
    x=pivots(d); highs=x.loc[x.swing_high,['date','high']].tail(8); lows=x.loc[x.swing_low,['date','low']].tail(8); last=float(x.close.iloc[-1]);
    trend='BULLISH' if last>x.ema50.iloc[-1]>x.ema200.iloc[-1] else 'BEARISH' if last<x.ema50.iloc[-1]<x.ema200.iloc[-1] else 'RANGE'
    sh=float(highs.high.iloc[-1]) if len(highs) else np.nan; sl=float(lows.low.iloc[-1]) if len(lows) else np.nan
    bsl=float(highs.high.max()) if len(highs) else np.nan; ssl=float(lows.low.min()) if len(lows) else np.nan
    bos='NONE'; choch='NONE'
    prev_high=float(highs.high.iloc[-2]) if len(highs)>=2 else np.nan; prev_low=float(lows.low.iloc[-2]) if len(lows)>=2 else np.nan
    if np.isfinite(prev_high) and last>prev_high: bos='BULLISH BOS'
    elif np.isfinite(prev_low) and last<prev_low: bos='BEARISH BOS'
    # structure shift against current trend
    if trend=='BEARISH' and np.isfinite(prev_high) and last>prev_high: choch='BULLISH CHoCH'
    elif trend=='BULLISH' and np.isfinite(prev_low) and last<prev_low: choch='BEARISH CHoCH'
    return x,{'trend':trend,'last_swing_high':sh,'last_swing_low':sl,'bsl':bsl,'ssl':ssl,'bos':bos,'choch':choch}

def fvgs(d,limit=6):
    x=d.reset_index(drop=True); zones=[]
    for i in range(2,len(x)):
        if x.low.iloc[i]>x.high.iloc[i-2]: zones.append({'type':'Bullish FVG','low':float(x.high.iloc[i-2]),'high':float(x.low.iloc[i]),'i':i})
        if x.high.iloc[i]<x.low.iloc[i-2]: zones.append({'type':'Bearish FVG','low':float(x.high.iloc[i]),'high':float(x.low.iloc[i-2]),'i':i})
    return zones[-limit:]

def spot_setup(d,s):
    x=d.iloc[-1]; price=float(x.close); atr=float(x.atr) if np.isfinite(x.atr) else price*0.03; support=s['last_swing_low']
    trend=s['trend']; momentum=float(x.rsi) if np.isfinite(x.rsi) else 50; vol=float(x.vol_ratio) if np.isfinite(x.vol_ratio) else 1
    reasons=[]
    if trend=='BULLISH': reasons.append('price is above the 50/200 EMA trend structure')
    if x.macd_hist>0: reasons.append('MACD momentum is positive')
    if 50<=momentum<=70: reasons.append('RSI is in a constructive momentum zone')
    if vol>=1.2: reasons.append('volume is above its 20-period average')
    # setup priority: sweep/reclaim -> breakout -> pullback
    setup='WAIT'; entry=price; stop=np.nan; tp1=np.nan; tp2=np.nan; tp3=np.nan
    if trend=='BULLISH' and np.isfinite(support):
        sweep=bool(x.low < support and x.close > support)
        breakout=bool(np.isfinite(s['bsl']) and price>s['bsl'] and vol>=1.0)
        pullback=bool(price<=x.ema20*1.02 and price>=x.ema20*0.98 and x.macd_hist>0)
        if sweep:
            setup='SPOT LONG — LIQUIDITY SWEEP + RECLAIM'; entry=price; stop=min(support, float(x.low))-0.5*atr
        elif breakout:
            setup='SPOT LONG — BREAKOUT'; entry=price; stop=min(float(x.ema20),support)-0.5*atr
        elif pullback:
            setup='SPOT LONG — EMA PULLBACK'; entry=price; stop=min(float(x.ema50),support)-0.5*atr
        if setup!='WAIT':
            risk=max(entry-stop,atr*0.5); tp1=entry+1.5*risk; tp2=entry+2.5*risk; tp3=entry+4*risk
            if price>entry+1.0*atr: setup='WAIT — price extended from reference entry'
    return {'setup':setup,'entry':entry,'stop':stop,'tp1':tp1,'tp2':tp2,'tp3':tp3,'reasons':reasons,'risk_pct':(entry-stop)/entry*100 if np.isfinite(stop) and entry>0 else np.nan}

# ---------------- ML + prediction engine ----------------
FEATURES=['rsi','macd_hist','atr_pct','stoch_k','adx','ret7','ret30','vol_ratio','price_ema50','price_ema200']

def _model_features(d):
    x=d.copy()
    x['price_ema50']=x.close/x.ema50-1
    x['price_ema200']=x.close/x.ema200-1
    return x

def _horizon_model(d, horizon):
    x=_model_features(d)
    future=(x.close.shift(-horizon)/x.close-1)*100
    x[f'future_{horizon}']=future
    x[f'target_{horizon}']=(future>0).astype(int)
    work=x.dropna(subset=FEATURES+[f'target_{horizon}',f'future_{horizon}']).copy()
    if len(work)<180:
        return {'status':'insufficient','rows':len(work),'horizon':horizon}
    split=int(len(work)*0.75)
    tr=work.iloc[:split].copy(); te=work.iloc[split:].copy()
    if len(te)<25 or tr[f'target_{horizon}'].nunique()<2:
        return {'status':'insufficient','rows':len(work),'horizon':horizon}
    Xtr=tr[FEATURES]; Xte=te[FEATURES]
    ytr=tr[f'target_{horizon}'].astype(int); yte=te[f'target_{horizon}'].astype(int)
    lr=Pipeline([('scale',StandardScaler()),('m',LogisticRegression(max_iter=2000))])
    rf=RandomForestClassifier(n_estimators=350,max_depth=7,min_samples_leaf=5,random_state=42,n_jobs=-1,class_weight='balanced')
    lr.fit(Xtr,ytr); rf.fit(Xtr,ytr)
    pl=lr.predict_proba(Xte)[:,1]; pr=rf.predict_proba(Xte)[:,1]; pe=(pl+pr)/2
    pred=(pe>=.5).astype(int)
    latest=x.dropna(subset=FEATURES).iloc[-1:][FEATURES]
    p_lr=float(lr.predict_proba(latest)[0,1]); p_rf=float(rf.predict_proba(latest)[0,1]); p=(p_lr+p_rf)/2
    reg=RandomForestRegressor(n_estimators=350,max_depth=7,min_samples_leaf=5,random_state=42,n_jobs=-1)
    reg.fit(Xtr,tr[f'future_{horizon}'])
    exp=float(reg.predict(latest)[0])
    tree_preds=np.array([est.predict(latest)[0] for est in reg.estimators_],dtype=float)
    spread=float(np.std(tree_preds)) if len(tree_preds)>1 else 0.0
    residual=te[f'future_{horizon}'].to_numpy()-reg.predict(Xte)
    residual_std=float(np.std(residual,ddof=1)) if len(residual)>1 else spread
    model_range=max(spread,residual_std)
    imp=pd.DataFrame({'feature':FEATURES,'importance':rf.feature_importances_}).sort_values('importance',ascending=False)
    label='BULLISH' if p>=.58 else 'BEARISH' if p<=.42 else 'NEUTRAL'
    return {
        'status':'ok','rows':len(work),'horizon':horizon,
        'metrics':{'accuracy':accuracy_score(yte,pred),'precision':precision_score(yte,pred,zero_division=0),'recall':recall_score(yte,pred,zero_division=0),'f1':f1_score(yte,pred,zero_division=0)},
        'p_lr':p_lr,'p_rf':p_rf,'p':p,'label':label,
        'expected_return':exp,'model_spread':model_range,'importance':imp
    }

def prediction_engine(d):
    out={}
    for h in [1,3,5]: out[h]=_horizon_model(d,h)
    return out

def train_ai(d):
    # Backward-compatible summary used by the main dashboard.
    preds=prediction_engine(d)
    p5=preds[5]
    if p5['status']!='ok': return {'status':'insufficient','rows':p5.get('rows',0),'predictions':preds}
    return {
        'status':'ok','rows':p5['rows'],'metrics':p5['metrics'],
        'p_lr':p5['p_lr'],'p_rf':p5['p_rf'],'p':p5['p'],
        'expected5':p5['expected_return'],'importance':p5['importance'],
        'predictions':preds
    }

# ---------------- charts ----------------
def money(v):
    if v is None or not np.isfinite(v): return '—'
    if abs(v)>=1e9:return f'${v/1e9:.2f}B'
    if abs(v)>=1e6:return f'${v/1e6:.2f}M'
    if abs(v)>=1e3:return f'${v/1e3:.2f}K'
    if abs(v)>=1:return f'${v:,.2f}'
    if abs(v)>=.01:return f'${v:.4f}'
    return f'${v:.8f}'

def fmt(v): return '—' if not np.isfinite(v) else f'{v:.2f}'

def main_chart(d,name,setup,s):
    x=d.tail(220).copy(); fig=go.Figure();
    fig.add_trace(go.Candlestick(x=x.date,open=x.open,high=x.high,low=x.low,close=x.close,name='Spot price',increasing_line_color='#16a34a',decreasing_line_color='#dc2626'))
    for col,label in [('ema20','EMA 20'),('ema50','EMA 50'),('ema200','EMA 200')]: fig.add_trace(go.Scatter(x=x.date,y=x[col],name=label,line={'width':1.4}))
    # swing points / liquidity
    px=x[x.swing_high]; py=x[x.swing_low]
    if len(px): fig.add_trace(go.Scatter(x=px.date,y=px.high,mode='markers+text',text=['BSL']*len(px),textposition='top center',name='Buy-side liquidity',marker={'size':7}))
    if len(py): fig.add_trace(go.Scatter(x=py.date,y=py.low,mode='markers+text',text=['SSL']*len(py),textposition='bottom center',name='Sell-side liquidity',marker={'size':7}))
    if np.isfinite(s['bsl']): fig.add_hline(y=s['bsl'],line_dash='dash',annotation_text=f"BSL {money(s['bsl'])}")
    if np.isfinite(s['ssl']): fig.add_hline(y=s['ssl'],line_dash='dash',annotation_text=f"SSL {money(s['ssl'])}")
    # FVG zones
    for z in fvgs(d.tail(220)):
        if z['type']=='Bullish FVG': fig.add_hrect(y0=z['low'],y1=z['high'],fillcolor='green',opacity=.10,line_width=0)
        else: fig.add_hrect(y0=z['low'],y1=z['high'],fillcolor='red',opacity=.08,line_width=0)
    if setup['setup'].startswith('SPOT LONG'):
        for val,label in [(setup['entry'],'ENTRY'),(setup['stop'],'STOP'),(setup['tp1'],'TP1'),(setup['tp2'],'TP2'),(setup['tp3'],'TP3')]:
            if np.isfinite(val): fig.add_hline(y=val,line_dash='dot',annotation_text=f'{label} {money(val)}')
    fig.update_layout(height=680,template='plotly_white',xaxis_rangeslider_visible=False,hovermode='x unified',margin={'l':10,'r':10,'t':35,'b':10},legend={'orientation':'h'})
    return fig

def momentum_chart(d):
    x=d.tail(220); fig=go.Figure(); fig.add_trace(go.Scatter(x=x.date,y=x.rsi,name='RSI',line={'width':2})); fig.add_hline(y=70,line_dash='dot'); fig.add_hline(y=30,line_dash='dot'); fig.add_trace(go.Scatter(x=x.date,y=x.adx,name='ADX')); fig.update_layout(height=330,template='plotly_white',hovermode='x unified'); return fig

def volume_chart(d):
    x=d.tail(220); fig=go.Figure(); fig.add_trace(go.Bar(x=x.date,y=x.volume,name='Volume')); fig.add_trace(go.Scatter(x=x.date,y=x.vol_ma,name='Volume MA20')); fig.update_layout(height=300,template='plotly_white',hovermode='x unified'); return fig

# ---------------- prediction chart ----------------
def prediction_chart(d,preds):
    price=float(d.close.iloc[-1]); now=d.date.iloc[-1]
    xs=[now]
    ys=[price]
    labels=['Now']
    for h in [1,3,5]:
        z=preds[h]
        if z.get('status')=='ok':
            xs.append(now+pd.Timedelta(days=h)); ys.append(price*(1+z['expected_return']/100)); labels.append(f'{h}D')
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=xs,y=ys,mode='lines+markers+text',text=labels,textposition='top center',name='Model projected price',line={'width':3}))
    fig.add_hline(y=price,line_dash='dot',annotation_text=f'Current {money(price)}')
    fig.update_layout(height=420,template='plotly_white',hovermode='x unified',xaxis_rangeslider_visible=False,margin={'l':10,'r':10,'t':35,'b':10})
    return fig

# ---------------- UI ----------------
st.title('🧠 AI Crypto Research Terminal V5.2')
st.caption('TradingView-style research • multi-timeframe structure • liquidity • FVG • spot setups • prediction • AI • risk • NO TRADE')
with st.sidebar:
    st.header('⚙️ Market data')
    if st.button('🔄 Sync coin catalogue',use_container_width=True):
        try: st.session_state['n']=sync_catalog(); st.success(f"Saved {st.session_state['n']:,} coins")
        except Exception as e: st.error(f'Catalogue sync failed: {e}')
    st.metric('Local coin catalogue',f"{db().execute('SELECT COUNT(*) FROM coin_catalog').fetchone()[0]:,}")
    st.divider(); st.header('Chart')
    tf=st.selectbox('Main timeframe',['1W','2D','1D','4h','1h'],index=2)
    history=st.selectbox('History',['730 days','365 days','180 days'],index=0)
    st.checkbox('Show liquidity labels',True); st.checkbox('Show FVG zones',True)
    st.divider(); st.caption('Spot research only. This version creates analysis and hypothetical spot setups; it does not place orders.')

if db().execute('SELECT COUNT(*) FROM coin_catalog').fetchone()[0]==0:
    st.warning('Coin catalogue is empty. Click Sync coin catalogue.'); st.stop()

q=st.text_input('🔎 Search any coin',value='btc'); results=search_catalog(q,100)
if results.empty: st.error('No coin found.'); st.stop()
opts=[f"{r.name} ({str(r.symbol).upper()}) — {r.id}" for r in results.itertuples()]; choice=st.selectbox('Select coin',opts); sel=results.iloc[opts.index(choice)]
run=st.button('🚀 RUN FULL MULTI-TIMEFRAME SPOT ANALYSIS',type='primary',use_container_width=True)
key=f"{sel.id}:{tf}:{history}"
if run or st.session_state.get('key')!=key:
    try:
        with st.spinner('Collecting market data, mapping structure, scanning liquidity and training AI...'):
            base,source=load_base(sel,int(history.split()[0])); base=add_indicators(base); base,st1=structure(base); setup=spot_setup(base,st1); ai=train_ai(base)
            mt={}
            for t in ['1W','2D','1D','4h','1h']:
                td,src=timeframe_data(sel,t); td=add_indicators(td); td,ss=structure(td); mt[t]={'data':td,'source':src,'s':ss,'last':td.iloc[-1]}
            st.session_state['result']={'base':base,'source':source,'structure':st1,'setup':setup,'ai':ai,'mt':mt,'key':key,'name':str(sel['name']),'symbol':str(sel.symbol).upper()}
            st.session_state.pop('error',None)
    except Exception as e:
        st.session_state.pop('result',None); st.session_state['error']=str(e)

if st.session_state.get('error'):
    st.error('Analysis could not be completed'); st.code(st.session_state['error']); st.info('Provider order: Binance Spot → CoinGecko → CoinPaprika daily fallback. If CoinGecko returns 401, the engine automatically tries CoinPaprika for up to 365 days. No fake market data is generated.')
R=st.session_state.get('result')
if R:
    d=R['base']; s=R['structure']; setup=R['setup']; ai=R['ai']; last=d.iloc[-1]
    st.subheader(f"{R['name']} ({R['symbol']})")
    a,b,c,e,f=st.columns(5); a.metric('Spot price',money(last.close)); b.metric('Trend',s['trend']); c.metric('RSI',fmt(last.rsi)); e.metric('7D',f"{fmt(last.ret7)}%"); f.metric('AI UP',f"{ai['p']*100:.1f}%" if ai['status']=='ok' else 'N/A')
    tabs=st.tabs(['📊 Market Report','🕯️ TradingView-style chart','🧠 Smart Money','🎯 Spot Trade Setup','🔮 Prediction','🤖 AI','🔭 Multi-Timeframe','📚 Advanced Data'])
    with tabs[0]:
        st.markdown('### What is happening?')
        st.write(f"**Trend:** {s['trend']} • **BOS:** {s['bos']} • **CHoCH:** {s['choch']} • **Volume ratio:** {fmt(last.vol_ratio)}x")
        st.markdown('### Why?')
        bullets=[]
        if last.close>last.ema20: bullets.append('Price is above EMA20, showing positive short-term positioning.')
        if last.ema20>last.ema50: bullets.append('EMA20 is above EMA50, supporting the short/medium trend.')
        if last.close>last.ema200: bullets.append('Price is above EMA200, supporting the long-term trend.')
        if last.macd_hist>0: bullets.append('MACD histogram is positive, so momentum is currently upward.')
        if 50<=last.rsi<=70: bullets.append('RSI is in a constructive zone rather than an extreme.')
        if last.vol_ratio>=1.2: bullets.append('Volume is above its 20-period average, adding participation evidence.')
        for z in bullets or ['Signals are mixed; no single factor dominates.']: st.write('•',z)
        st.markdown('### What could prove this wrong?')
        st.write(f"A sustained break below the latest swing low / SSL around **{money(s['ssl'])}** would weaken the current structure. A failure at nearby resistance/BSL around **{money(s['bsl'])}** would also reduce bullish confidence.")
        st.markdown('### Research state')
        st.info('🟢 Spot setup detected' if setup['setup'].startswith('SPOT LONG') else '🟡 NO TRADE / WAIT — conditions are not sufficiently aligned for a defined spot setup.')
    with tabs[1]:
        st.plotly_chart(main_chart(d,R['name'],setup,s),use_container_width=True)
        st.plotly_chart(momentum_chart(d),use_container_width=True); st.plotly_chart(volume_chart(d),use_container_width=True)
    with tabs[2]:
        st.markdown('### Liquidity map')
        c1,c2,c3=st.columns(3); c1.metric('BSL',money(s['bsl'])); c2.metric('SSL',money(s['ssl'])); c3.metric('Structure',s['trend'])
        st.write('**BSL (buy-side liquidity):** prior swing-high area where stops/liquidity may cluster. **SSL (sell-side liquidity):** prior swing-low area. These are hypotheses derived from price structure, not guaranteed order-book levels.')
        st.write('**BOS:** break of a prior swing level. **CHoCH:** a potential change in structure. **FVG:** a three-candle price imbalance zone.')
        z=pd.DataFrame(fvgs(d.tail(250))); st.dataframe(z[['type','low','high']] if not z.empty else pd.DataFrame({'FVG':['No recent gap detected']}),use_container_width=True,hide_index=True)
    with tabs[3]:
        st.subheader(setup['setup'])
        if setup['setup'].startswith('SPOT LONG'):
            cols=st.columns(5); cols[0].metric('Entry',money(setup['entry'])); cols[1].metric('Stop',money(setup['stop'])); cols[2].metric('TP1',money(setup['tp1'])); cols[3].metric('TP2',money(setup['tp2'])); cols[4].metric('TP3',money(setup['tp3']))
            rr1=(setup['tp1']-setup['entry'])/(setup['entry']-setup['stop']); rr2=(setup['tp2']-setup['entry'])/(setup['entry']-setup['stop']); rr3=(setup['tp3']-setup['entry'])/(setup['entry']-setup['stop'])
            st.write(f"**Risk:** {setup['risk_pct']:.2f}% from entry to reference stop • **R:R:** 1:{rr1:.2f} / 1:{rr2:.2f} / 1:{rr3:.2f}")
            for r in setup['reasons']: st.write('•',r)
            st.warning('This is a hypothetical research setup for spot markets. It is not an order and the app does not execute trades.')
        else:
            st.warning('NO TRADE / WAIT'); st.write('The engine did not find enough alignment for a defined spot-long setup. That is an intentional outcome, not an error.')
    with tabs[4]:
        st.subheader('🔮 AI price-direction & return prediction')
        st.caption('The model estimates future direction and return from historical patterns. These are research estimates, not guaranteed future prices.')
        preds=ai.get('predictions',{})
        rows=[]
        current=float(last.close)
        for h in [1,3,5]:
            z=preds.get(h,{})
            if z.get('status')=='ok':
                exp=float(z['expected_return']); projected=current*(1+exp/100); spread=float(z.get('model_spread',0)); low=current*(1+(exp-spread)/100); high=current*(1+(exp+spread)/100)
                rows.append({'Horizon':f'{h} day' if h==1 else f'{h} days','Direction':z['label'],'UP probability':f"{z['p']*100:.1f}%",'Expected return':f"{exp:+.2f}%",'Model projected price':money(projected),'Model range':f"{money(low)} – {money(high)}",'Holdout accuracy':f"{z['metrics']['accuracy']*100:.1f}%"})
            else:
                rows.append({'Horizon':f'{h} day' if h==1 else f'{h} days','Direction':'N/A','UP probability':'N/A','Expected return':'N/A','Model projected price':'N/A','Model range':'N/A','Holdout accuracy':'N/A'})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
        if any(preds.get(h,{}).get('status')=='ok' for h in [1,3,5]):
            st.plotly_chart(prediction_chart(d,preds),use_container_width=True)
            z=preds.get(5,{})
            if z.get('status')=='ok':
                st.markdown('### 5-day prediction in plain English')
                if z['label']=='BULLISH':
                    st.success(f"The models currently lean **UP** over the next 5 days, with an ensemble UP probability of **{z['p']*100:.1f}%** and an estimated return of **{z['expected_return']:+.2f}%**.")
                elif z['label']=='BEARISH':
                    st.error(f"The models currently lean **DOWN** over the next 5 days, with an ensemble UP probability of **{z['p']*100:.1f}%** and an estimated return of **{z['expected_return']:+.2f}%**.")
                else:
                    st.warning(f"The models are **NEUTRAL** over the next 5 days. UP probability is **{z['p']*100:.1f}%** and estimated return is **{z['expected_return']:+.2f}%**.")
                st.write('The projected price is a model output derived from the current price and expected return. The range reflects model/residual spread, not a guaranteed confidence interval.')
                st.markdown('### Prediction + spot setup alignment')
                if setup['setup'].startswith('SPOT LONG') and z['label']=='BULLISH' and z['expected_return']>0:
                    st.success('Prediction and the current spot setup point in the same direction. This is still a research setup, not a guaranteed trade.')
                elif setup['setup'].startswith('SPOT LONG') and z['label']!='BEARISH':
                    st.info('The spot setup exists, but the prediction is not strongly confirming it. Treat the setup cautiously.')
                else:
                    st.info('No confirmed spot setup from the structure engine. Prediction alone does not create a trade.')
        else:
            st.warning(f"Prediction models skipped safely: only {ai.get('rows',0)} clean training rows were available.")

    with tabs[5]:
        if ai['status']=='ok':
            a,b,c,d1=st.columns(4); a.metric('Holdout accuracy',f"{ai['metrics']['accuracy']*100:.1f}%"); b.metric('F1',f"{ai['metrics']['f1']*100:.1f}%"); c.metric('RF UP',f"{ai['p_rf']*100:.1f}%"); d1.metric('Expected 5D',f"{ai['expected5']:+.2f}%")
            st.write('Ensemble UP probability:',f"{ai['p']*100:.1f}%")
            st.dataframe(ai['importance'],use_container_width=True,hide_index=True)
            st.caption('The holdout is chronological. Model probability is a research statistic, not a guarantee.')
        else: st.warning(f"AI skipped safely: only {ai['rows']} clean rows were available.")
    with tabs[6]:
        rows=[]
        for t,v in R['mt'].items(): rows.append({'Timeframe':t,'Trend':v['s']['trend'],'BOS':v['s']['bos'],'CHoCH':v['s']['choch'],'RSI':round(float(v['last'].rsi),1) if np.isfinite(v['last'].rsi) else np.nan,'7D/period return':round(float(v['last'].ret7),2) if np.isfinite(v['last'].ret7) else np.nan,'Source':v['source']})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
        st.write('A high-quality spot setup should ideally have higher-timeframe structure and lower-timeframe entry logic that do not directly conflict.')
    with tabs[7]:
        cols=['date','open','high','low','close','volume','ema20','ema50','ema200','rsi','macd_hist','atr_pct','adx','vol_ratio','ret7','ret30']
        st.dataframe(d[cols].tail(200).sort_values('date',ascending=False),use_container_width=True,hide_index=True)
        st.download_button('⬇️ Download research CSV',d.to_csv(index=False).encode(),file_name=f"{sel.id}_spot_research.csv",mime='text/csv')

st.divider(); st.caption('Research-only. Spot setup means hypothetical buy-side analysis; no automated order execution. Market data can be delayed, unavailable, or rate-limited.')
