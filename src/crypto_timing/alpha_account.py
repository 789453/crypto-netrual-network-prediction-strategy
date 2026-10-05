"""One net position per coin, self-financing shares, real funding and actual turnover."""
from dataclasses import asdict
import numpy as np
from .alpha_policy import decide,covariance,project

def account(price,signal,funding_cash,expected_funding,dates,hours,policy,fee=.0004,minute=None,funding_minute=None,mode='economic',retain=False,decision_adapter=None):
    assert len(hours)==len(signal) and np.all(np.diff(hours)==1),'continuous prediction clocks required; missing forecasts cannot be silently bridged'
    minute_peak=1.;minute_drawdown=0.
    ns=price.shape[1];qty=np.zeros(ns);cash=1.;age=np.zeros(ns,int);nav=[1.];gross=[];net=[];margin=[];turn=[];notional=[];fees=[];fund=[];pnl=[];weights=[];trades=[];episodes=[];held_from=np.full(ns,-1,int);entry_price=np.zeros(ns);entry_qty=np.zeros(ns);episode_gain=np.zeros(ns);episode_cost=np.zeros(ns);episode_fund=np.zeros(ns);longp=[];shortp=[];breach=0;bankrupt=False;reasons={};minimum_equity=1.
    # Cache past-only covariance snapshots; same rules for every model.
    cov,beta=covariance(price,max(0,int(hours[0])-1))
    for row,h in enumerate(hours):
        p=price[h];equity=cash+float(qty@p)
        if equity<=0:bankrupt=True;break
        if h%24==0 or row==0:cov,beta=covariance(price,max(0,h-1))
        old=qty*p/equity
        if decision_adapter is not None:
            if mode!='economic':raise ValueError('decision_adapter requires economic mode')
            desired,change,reason=decision_adapter(row,signal[row],cov,beta,old,age,policy)
        elif mode=='economic':desired,change,reason=decide(signal[row],expected_funding[h],cov,beta,old,age,policy)
        elif mode=='market':
            w=np.ones(ns)*policy.cap/ns;vol=np.sqrt(w@cov@w)*np.sqrt(24*365.25);w*=min(1,policy.annual_risk/max(vol,1e-9));desired=w;change=(age>=policy.horizon)|(qty==0);reason=np.full(ns,'market_rebalance',object)
        elif mode=='phase':
            # Unified old-policy comparator: 4h rolling levels, exact hourly target.
            desired=np.tanh(signal[max(0,row-3):row+1].mean(0)/.001)/ns*policy.cap;desired=project(desired,policy.cap,policy.cap/2,policy.coin_cap);change=np.ones(ns,bool);reason=np.full(ns,'phase_rebalance',object)
        elif mode=='full_rank':
            order=np.argsort(signal[row]);w=np.zeros(ns);w[order[:ns//2]]=-policy.cap/ns;w[order[ns//2:]]=policy.cap/ns;desired=project(w,policy.cap,policy.cap/2,policy.coin_cap);change=(age>=policy.horizon)|(qty==0);reason=np.full(ns,'forced_rank',object)
        else:raise ValueError(mode)
        if mode in ('market','full_rank'):age=np.where(change&(age>=policy.horizon),0,age)
        # Solve post-fee equity for executable target shares. Unchanged holdings keep exact shares.
        after=equity
        for _ in range(8):
            target=np.where(change,desired*after/p,qty);amount=np.abs(target-qty)*p;after=equity-fee*amount.sum()
        delta=target-qty;amount=np.abs(delta)*p;cost=fee*amount
        # Closing/reversing shares ends a position episode. Partial changes retain its age/basis.
        for j in np.flatnonzero(np.abs(delta)>1e-12):
            closing=qty[j]!=0 and (target[j]==0 or np.sign(target[j])!=np.sign(qty[j]))
            if closing:
                episodes.append({'symbol_id':int(j),'start_hour':int(held_from[j]),'end_hour':int(h),'holding_hours':int(h-held_from[j]),'side':int(np.sign(qty[j])),'price_pnl':float(episode_gain[j]),'fee':float(episode_cost[j]+fee*abs(qty[j])*p[j]),'funding':float(episode_fund[j]),'exit_reason':str(reason[j])})
                episode_gain[j]=episode_cost[j]=episode_fund[j]=0
            opening=target[j]!=0 and (qty[j]==0 or np.sign(target[j])!=np.sign(qty[j]))
            if opening:held_from[j]=h;age[j]=0;entry_price[j]=p[j];entry_qty[j]=target[j]
            episode_cost[j]+=cost[j] if not closing else max(0,cost[j]-fee*abs(qty[j])*p[j])
            reasons[str(reason[j])]=reasons.get(str(reason[j]),0)+1
            if retain:trades.append({'hour':int(h),'symbol_id':int(j),'price':float(p[j]),'quantity_before':float(qty[j]),'quantity_after':float(target[j]),'change':float(delta[j]),'notional':float(amount[j]),'fee':float(cost[j]),'equity_before':float(equity),'reason':str(reason[j])})
        cash-=float(delta@p+cost.sum());qty=target
        post=cash+float(qty@p);w=qty*p/post;gross.append(float(np.abs(w).sum()));net.append(float(w.sum()));margin.append(float(np.abs(w).sum()/5));weights.append(w.copy())
        price_gain=qty*(price[h+1]-p);funding=-qty*funding_cash[h]
        if minute is not None:
            path=np.concatenate((np.asarray(minute[h],float),price[h+1][None]),0)
            ee=cash+path@qty
            if funding_minute is not None:ee-=np.cumsum(funding_minute[h],axis=0)@qty
            low=float(ee.min());minimum_equity=min(minimum_equity,low)
            peaks=np.maximum.accumulate(np.concatenate(([minute_peak],ee)))[1:];minute_drawdown=min(minute_drawdown,float(np.min(ee/peaks-1)));minute_peak=max(minute_peak,float(ee.max()))
            leverage=np.abs(path*qty).sum(1)/np.maximum(ee,1e-12)
            breach+=int(np.any(leverage>policy.cap*1.05))
            if low<=0:bankrupt=True
        cash+=float(funding.sum());end=cash+float(qty@price[h+1])
        episode_gain+=price_gain;episode_fund+=funding
        nav.append(end);turn.append(float(amount.sum()/equity));notional.append(amount.copy());fees.append(cost.copy());fund.append(funding.copy());pnl.append(price_gain.copy());longp.append(np.where(qty>0,price_gain,0));shortp.append(np.where(qty<0,price_gain,0));age+=qty!=0
        if bankrupt:break
    # Actual terminal close, including fee; does not silently disappear at model switches.
    if len(pnl):
        last=hours[len(pnl)-1]+1;p=price[last];terminal=fee*np.abs(qty)*p;closing_equity=nav[-1];cash+=float(qty@p-terminal.sum());fees[-1]+=terminal;notional[-1]+=np.abs(qty)*p;nav[-1]=cash;turn[-1]+=float((np.abs(qty)*p).sum()/max(nav[-1]+terminal.sum(),1e-12))
        for j in np.flatnonzero(qty):
            episodes.append({'symbol_id':int(j),'start_hour':int(held_from[j]),'end_hour':int(last),'holding_hours':int(last-held_from[j]),'side':int(np.sign(qty[j])),'price_pnl':float(episode_gain[j]),'fee':float(episode_cost[j]+terminal[j]),'funding':float(episode_fund[j]),'exit_reason':'terminal'})
            if retain:trades.append({'hour':int(last),'symbol_id':int(j),'price':float(p[j]),'quantity_before':float(qty[j]),'quantity_after':0.,'change':float(-qty[j]),'notional':float(abs(qty[j])*p[j]),'fee':float(terminal[j]),'equity_before':float(closing_equity),'reason':'terminal'})
    nav=np.asarray(nav);dd=nav/np.maximum.accumulate(nav)-1
    daily=[]
    used=hours[:len(pnl)];days=dates[used].astype('datetime64[D]')
    for d in np.unique(days):k=np.flatnonzero(days==d);daily.append(nav[k[-1]+1]/nav[k[0]]-1)
    daily=np.asarray(daily);vol=float(daily.std()*np.sqrt(365.25));annual=float(nav[-1]**(365.25/max(len(daily),1))-1) if nav[-1]>0 else -1.
    result={'return':float(nav[-1]-1),'max_drawdown':float(dd.min()),'daily_sharpe':float(daily.mean()/max(daily.std(),1e-12)*np.sqrt(365.25)),'annual_return':annual,'annual_vol':vol,'average_gross':float(np.mean(gross)) if gross else 0.,'average_net':float(np.mean(net)) if net else 0.,'gross_utilization':float(np.mean(gross)/policy.cap) if gross else 0.,'average_margin_fraction_at_5x':float(np.mean(margin)) if margin else 0.,'turnover':float(np.sum(turn)),'total_traded_notional_initial_equity_units':float(np.sum(notional)),'fee':float(np.sum(fees)),'funding':float(np.sum(fund)),'price_pnl':float(np.sum(pnl)),'price_bps_per_turnover':float(np.sum(pnl)/max(np.sum(notional),1e-12)*1e4),'holding_median':float(np.median([e['holding_hours'] for e in episodes])) if episodes else 0.,'holding_mean':float(np.mean([e['holding_hours'] for e in episodes])) if episodes else 0.,'participation':float(np.mean(np.asarray(gross)>1e-6)) if gross else 0.,'episodes':len(episodes),'minute_gross_limit_breach_hours':breach,'minimum_minute_equity':minimum_equity,'bankrupt':bankrupt,'exit_reasons':reasons,'policy':asdict(policy),'mode':mode,'fee_per_side':fee,'hours':len(pnl)}
    result['minute_max_drawdown']=minute_drawdown if minute is not None else None
    arrays={'nav':nav,'dates':dates[used],'hours':used,'weights':np.asarray(weights),'pnl':np.asarray(pnl),'funding':np.asarray(fund),'fees':np.asarray(fees),'turnover':np.asarray(turn),'traded_notional':np.asarray(notional),'long_pnl':np.asarray(longp),'short_pnl':np.asarray(shortp),'daily_returns':daily}
    return result,arrays,episodes,trades
