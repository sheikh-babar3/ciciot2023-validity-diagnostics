import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
RES='res'; OUT='ms/Closed-Form_Validity_Diagnostics_Revised/figures'
d=pd.read_csv(f'{RES}/per_fold_results.csv')
ar=pd.read_csv(f'{RES}/avg_ranks.csv',index_col=0).iloc[:,0].sort_values()
CD=13.2597
lab={'A1 Transformer':'Transformer (A1)','A2 IAT-DNN':'IAT-DNN (A2)','A4 GIDS':'GIDS (A4)','A5 GCN-DQN':'GCN-DQN (A5)'}
# ---------- 1. CD diagram (Demsar-style) ----------
fig,ax=plt.subplots(figsize=(10,4.4),dpi=200)
ax.set_xlim(-8.6,29.4); ax.set_ylim(-5.9,2.3); ax.axis('off')
ax.plot([1,20],[0,0],'k',lw=1.6)
for t in range(1,21):
    ax.plot([t,t],[0,0.15],'k',lw=1); ax.text(t,0.3,str(t),ha='center',va='bottom',fontsize=10)
ax.text(10.5,1.95,'Average Friedman rank on macro-F1 (lower is better)',ha='center',fontsize=11)
cut=ar.iloc[0]+CD
ax.plot([1,1+CD],[1.2,1.2],'k',lw=3); ax.text(1+CD/2,1.3,f'CD = {CD:.2f}',ha='center',fontsize=10.5,fontweight='bold')
ax.plot([cut,cut],[0,0.15],color='#a83232',lw=2.2)
ax.text(cut,0.95,f'1.00 + CD = {cut:.2f}',color='#a83232',fontsize=9,ha='center')
items=list(ar.items()); L=items[:10]; R=items[10:][::-1]
def draw(group,side):
    for i,(name,r) in enumerate(group):
        y=-0.9-i*0.5; red=r>cut; col='#a83232' if red else 'black'
        xl=0.4 if side=='L' else 20.6
        ax.plot([r,r,xl],[0,y,y],color='#555',lw=0.9)
        ax.text(xl-0.2 if side=='L' else xl+0.2,y,f'{lab.get(name,name)}  ({r:.2f})' if side=='L' else f'({r:.2f})  {lab.get(name,name)}',
                ha='right' if side=='L' else 'left',va='center',fontsize=10.5,color=col,fontweight='bold' if name=='CAFE-TF' else 'normal')
draw(L,'L'); draw(R,'R')
plt.savefig(f'{OUT}/Figure8_CD_diagram.png',bbox_inches='tight'); plt.close()
# ---------- 2. class distribution ----------
cnt={'DDoS':3100549,'DoS':737065,'Mirai':240118,'Normal':100294,'Spoofing':44580,'Recon':28920,'BruteForce_Other':4893,'Web':1910}
fig,ax=plt.subplots(figsize=(6.2,3.3),dpi=200)
cols=['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#8c564b','#7f7f7f','#bcbd22','#17becf']
b=ax.bar(list(cnt),list(cnt.values()),color=cols,edgecolor='k',lw=0.5); ax.set_yscale('log')
for x,v in zip(range(8),cnt.values()): ax.text(x,v*1.18,f'{v:,}',ha='center',fontsize=7)
ax.set_ylim(1e3,1.2e7); ax.set_ylabel('Rows (log scale)',fontsize=9)
ax.set_title('Regrouped 8-class taxonomy, 4,258,329 rows (imbalance 1,623:1)',fontsize=9.5,fontweight='bold')
plt.setp(ax.get_xticklabels(),rotation=35,ha='right',fontsize=8); ax.tick_params(axis='y',labelsize=8)
plt.tight_layout(); plt.savefig(f'{OUT}/DatasetExploration_class_distribution.png'); plt.close()
# ---------- 3. loss curves ----------
fig,axs=plt.subplots(1,3,figsize=(10,2.9),dpi=200)
for ax,(m,t) in zip(axs,[('CNN-1D','CNN-1D'),('MLP','MLP'),('CAFE-TF','CAFE-TF')]):
    for f in range(1,6):
        a=np.load(f'{RES}/loss_history/fold{f}_{m}.npy'); ax.plot(a,label=f'fold {f}',lw=1.2)
    ax.set_title(t,fontsize=10,fontweight='bold'); ax.set_xlabel('epoch',fontsize=9); ax.set_ylabel('training loss',fontsize=9); ax.tick_params(labelsize=8)
axs[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f'{OUT}/Perf_loss_curves.png'); plt.close()
# ---------- 4. accuracy vs macro-F1 ----------
g=d.groupby('model')[['accuracy','f1_macro']].agg(['mean','std'])
prop={'PANE','CAFE-TF','ARID','FedHAN','Cascaded Ensemble'}; lit={'A2 IAT-DNN','A4 GIDS','A5 GCN-DQN'}
short=lambda m: lab.get(m,m).replace(' (A1)','').replace(' (A2)','').replace(' (A4)','').replace(' (A5)','')
off={'PANE':(8,12),'A4 GIDS':(-30,22),'LSTM':(-34,-14),'Bi-LSTM':(6,-20),'A1 Transformer':(8,10),'VAE Detector':(8,-14),'Isolation Forest':(8,10),'One-Class SVM':(8,-12),
     'Cascaded Ensemble':(-100,6),'CAFE-TF':(-52,8),'A2 IAT-DNN':(-48,6)}
fig,ax=plt.subplots(figsize=(5.4,4.9),dpi=220)
for m,r in g.iterrows():
    c='#2E7D5B' if m in prop else '#E08E45' if m in lit else '#8C9BAB'
    x,y=r[('accuracy','mean')],r[('f1_macro','mean')]
    ax.errorbar(x,y,xerr=r[('accuracy','std')],yerr=r[('f1_macro','std')],fmt='o',color=c,ms=5.5,mec='k',mew=0.5,elinewidth=0.7,capsize=1.5)
    o=off.get(m,(7,6))
    ax.annotate(short(m),(x,y),xytext=o,textcoords='offset points',fontsize=8.6,arrowprops=dict(arrowstyle='-',color='#888',lw=0.5,shrinkA=0,shrinkB=2) if (abs(o[0])>18 or abs(o[1])>14) else None)
ax.axvline(0.7281,ls='--',color='grey',lw=1.1); ax.text(0.74,0.0,'always-predict-majority\naccuracy (0.728)',fontsize=8,color='dimgrey',va='bottom')
ax.set_xlim(-0.12,1.07); ax.set_ylim(-0.04,0.85); ax.grid(ls=':',alpha=0.5)
ax.set_xlabel('Accuracy (mean over folds; bars = std)',fontsize=10); ax.set_ylabel('Macro-F1 (mean over folds; bars = std)',fontsize=10); ax.tick_params(labelsize=9)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([],[],marker='o',ls='',color=c,mec='k',label=l) for c,l in [('#2E7D5B','Proposed'),('#E08E45','Literature'),('#8C9BAB','Baseline')]],loc='upper left',fontsize=9)
plt.tight_layout(); plt.savefig(f'{OUT}/Perf_accuracy_vs_macroF1.png'); plt.close()
# ---------- 5. fold box plot ----------
P=d.pivot(index='fold',columns='model',values='f1_macro'); order=P.mean().sort_values(ascending=False).index
fig,ax=plt.subplots(figsize=(5.8,4.0),dpi=220)
bp=ax.boxplot([P[m] for m in order],patch_artist=True,showmeans=True,widths=0.6,
    meanprops=dict(marker='^',markerfacecolor='#2ca02c',markeredgecolor='#2ca02c',markersize=4.5),medianprops=dict(color='#d95f02',lw=1.2),flierprops=dict(markersize=3.5))
for patch,m in zip(bp['boxes'],order):
    patch.set_facecolor('#2E7D5B' if m in prop else '#E08E45' if m in lit else '#8C9BAB'); patch.set_alpha(0.75)
ax.set_xticks(range(1,len(order)+1)); ax.set_xticklabels([short(m) for m in order],rotation=60,ha='right',fontsize=8.6)
ax.set_ylabel('Macro-F1 per fold',fontsize=10); ax.tick_params(axis='y',labelsize=9); ax.grid(axis='y',ls=':',alpha=0.5)
plt.tight_layout(); plt.savefig(f'{OUT}/Perf_fold_variance_boxplot.png'); plt.close()
