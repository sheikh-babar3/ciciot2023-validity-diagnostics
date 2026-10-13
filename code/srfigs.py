import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R='res'; O='sr/src/figures_new'
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans'})
d=pd.read_csv(f'{R}/per_fold_results.csv'); pc=pd.read_csv(f'{R}/per_class_metrics.csv')
lab={'A1 Transformer':'Transformer (A1)','A2 IAT-DNN':'IAT-DNN (A2)','A4 GIDS':'GIDS (A4)','A5 GCN-DQN':'GCN-DQN (A5)'}
L=lambda m:lab.get(m,m)
g=d.groupby('model').agg(f1=('f1_macro','mean'),f1s=('f1_macro','std'),acc=('accuracy','mean'),t=('time_sec','mean'))
order=g.f1.sort_values(ascending=False).index.tolist()
# A: per-class F1 heatmap
cls=['DDoS','DoS','Mirai','Normal','Spoofing','Recon','BruteForce_Other','Web']
M=pc.groupby(['model','cls']).f1.mean().unstack()[cls].loc[order]
fig,ax=plt.subplots(figsize=(7.2,6.4),dpi=220)
im=ax.imshow(M.values,cmap='viridis',vmin=0,vmax=1,aspect='auto')
ax.set_xticks(range(8)); ax.set_xticklabels(['DDoS','DoS','Mirai','Normal','Spoofing','Recon','BruteForce\n/Other','Web'],fontsize=8)
ax.set_yticks(range(len(order))); ax.set_yticklabels([L(m) for m in order],fontsize=8)
for i in range(M.shape[0]):
    for j in range(8):
        v=M.values[i,j]; ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=6.5,color='w' if v<0.6 else 'k')
ax.spines[:].set_visible(False); ax.xaxis.tick_top()
cb=fig.colorbar(im,ax=ax,fraction=0.035,pad=0.02); cb.set_label('Per-class F1 (mean of 5 folds)',fontsize=8)
plt.tight_layout(); plt.savefig(f'{O}/sr_perclass_heatmap.png'); plt.close()
# B: closed-form identity check
x=d.accuracy/8; fig,axs=plt.subplots(1,2,figsize=(7.2,3.3),dpi=220)
ax=axs[0]; ax.plot([0,.13],[0,.13],'k--',lw=0.8,label='identity  P = acc/K')
const=['A1 Transformer','VAE Detector','Isolation Forest']
for m,gr in d.groupby('model'):
    c=m in const
    ax.scatter(gr.accuracy/8,gr.precision,s=22 if c else 10,c='#c0392b' if c else '#7f8c8d',alpha=.9 if c else .45,edgecolor='none')
ax.set_xlabel('Predicted macro-precision  acc / K'); ax.set_ylabel('Observed macro-precision'); ax.set_title('(a) Precision identity',fontsize=9,loc='left')
ax.scatter([],[],c='#c0392b',s=22,label='constant-class models'); ax.scatter([],[],c='#7f8c8d',s=10,label='other models (100 points)'); ax.legend(fontsize=7,frameon=False,loc='upper left')
ax=axs[1]
for m,gr in d.groupby('model'):
    c=m in const
    ax.scatter(np.full(len(gr),0)+np.random.RandomState(abs(hash(m))%99).uniform(-.3,.3,len(gr))+(1 if c else 0),gr.recall,s=22 if c else 10,c='#c0392b' if c else '#7f8c8d',alpha=.9 if c else .45,edgecolor='none')
ax.axhline(.125,color='k',ls='--',lw=.8); ax.text(1.5,.133,'1/K = 0.125',fontsize=7,ha='right')
ax.set_xticks([0,1]); ax.set_xticklabels(['other models','constant-class\nmodels']); ax.set_ylabel('Observed macro-recall'); ax.set_ylim(0,1); ax.set_title('(b) Recall identity',fontsize=9,loc='left')
plt.tight_layout(); plt.savefig(f'{O}/sr_identity_check.png'); plt.close()
# C: cascade stage resolution
c=pd.read_csv(f'{R}/cascade_stage_resolution.csv')
fig,ax=plt.subplots(figsize=(3.6,3.0),dpi=220)
ax.bar(c.fold-.19,c.pct_stage2,.38,color='#f39c12',label='Stage 2 (CAFE-TF)'); ax.bar(c.fold+.19,c.pct_stage3,.38,color='#c0392b',label='Stage 3 (ARID/FedHAN)')
ax.set_ylim(0,3.6); ax.set_xlabel('Fold'); ax.set_ylabel('Share of test rows escalated (%)'); ax.legend(fontsize=6.5,frameon=False,loc='upper right',ncol=1)
ax.set_title(f'Stage 1 (PANE) resolves {c.pct_stage1.min():.1f}-{c.pct_stage1.max():.1f}%',fontsize=8,loc='left')
plt.tight_layout(); plt.savefig(f'{O}/sr_cascade_resolution.png'); plt.close()
# D: Nemenyi p
n=pd.read_csv(f'{R}/nemenyi_pvalues.csv',index_col=0); ar=pd.read_csv(f'{R}/avg_ranks.csv',index_col=0).iloc[:,0].sort_values().index
n=n.loc[ar,ar]; fig,ax=plt.subplots(figsize=(5.4,4.8),dpi=220)
im=ax.imshow(-np.log10(np.clip(n.values,1e-6,1)),cmap='magma_r',vmin=0,vmax=4)
ax.set_xticks(range(20)); ax.set_yticks(range(20)); ax.set_xticklabels([L(m) for m in ar],rotation=90,fontsize=6); ax.set_yticklabels([L(m) for m in ar],fontsize=6)
for i in range(20):
    for j in range(20):
        if n.values[i,j]<0.05 and i!=j: ax.text(j,i,'*',ha='center',va='center',fontsize=6,color='w')
cb=fig.colorbar(im,ax=ax,fraction=.04,pad=.02); cb.set_label(r'$-\log_{10}\,p$ (Nemenyi); * = p<0.05',fontsize=7)
plt.tight_layout(); plt.savefig(f'{O}/sr_nemenyi_heatmap.png'); plt.close()
# E: time vs macro-F1 with Pareto
fig,ax=plt.subplots(figsize=(6.2,3.8),dpi=220)
p=g.drop('Cascaded Ensemble').sort_values('t'); best=-1; fr=[]
for m,r in p.iterrows():
    if r.f1>best: best=r.f1; fr.append(m)
for m,r in g.iterrows():
    ax.scatter(r.t,r.f1,s=34,c='none' if m=='Cascaded Ensemble' else ('#c0392b' if m in fr else '#7f8c8d'),edgecolors='#2c3e50' if m=='Cascaded Ensemble' else 'none',linewidths=1.2,zorder=3)
    off={'LSTM':(-6,-10),'Bi-LSTM':(3,-11),'A4 GIDS':(6,4),'FedHAN':(3,5),'Linear SVM':(-10,-11)}.get(m,(3,3)); ax.annotate(L(m),(r.t,r.f1),fontsize=6.3,xytext=off,textcoords='offset points')
ax.plot([g.loc[m].t for m in fr],[g.loc[m].f1 for m in fr],color='#c0392b',lw=1,ls='--',zorder=2)
ax.set_xscale('log'); ax.set_xlabel('Mean wall-clock time per fold (s, log scale)'); ax.set_ylabel('Macro-F1'); ax.set_xlim(1.5,1.2e5)
plt.tight_layout(); plt.savefig(f'{O}/sr_pareto.png'); plt.close(); print(fr)
# F: ranked macro-F1 bars w/ SD
fig,ax=plt.subplots(figsize=(6.2,4.4),dpi=220)
gg=g.loc[order][::-1]; cols=['#c0392b' if m in const else '#2c7fb8' for m in gg.index]
ax.barh(range(20),gg.f1,xerr=gg.f1s,color=cols,error_kw=dict(lw=.8,capsize=1.5))
ax.set_yticks(range(20)); ax.set_yticklabels([L(m) for m in gg.index],fontsize=7)
ax.axvline(0.1053,color='k',ls=':',lw=.9); ax.plot([],[],'k:',lw=.9,label='constant-DDoS predictor (0.105)')
ax.set_xlabel('Macro-F1 (mean ± SD over 5 folds)'); ax.barh([],[])
ax.scatter([],[],c='#c0392b',marker='s',label='constant-class (non-classifying)'); ax.scatter([],[],c='#2c7fb8',marker='s',label='other'); ax.legend(fontsize=7,frameon=False,loc='lower right')
plt.tight_layout(); plt.savefig(f'{O}/sr_macroF1_ranked.png'); plt.close()
# G: rare-class F1 for top-4 methods by fold-free means (grouped bars)
top=['CAFE-TF','Cascaded Ensemble','PANE','A4 GIDS']; rc=['BruteForce_Other','Web','Recon','Spoofing','Normal']
Mx=pc.groupby(['model','cls']).f1.agg(['mean','std']).loc[top]
fig,ax=plt.subplots(figsize=(6.2,3.3),dpi=220); w=.2
for i,m in enumerate(top):
    ax.bar(np.arange(len(rc))+(i-1.5)*w,[Mx.loc[(m,c_),'mean'] for c_ in rc],w,yerr=[Mx.loc[(m,c_),'std'] for c_ in rc],label=L(m),error_kw=dict(lw=.7,capsize=1))
ax.set_xticks(range(len(rc))); ax.set_xticklabels(['BruteForce/Other','Web','Recon','Spoofing','Normal']); ax.set_ylabel('F1 (mean ± SD, 5 folds)'); ax.set_ylim(0,1.05); ax.legend(fontsize=7,frameon=False,ncol=4,loc='upper center')
plt.tight_layout(); plt.savefig(f'{O}/sr_rareclass.png'); plt.close()
print(M.round(2).to_string())
