import numpy as np
import pandas as pd
import glob
import os 
from pathlib import Path
from ppde.third_party.hsu import io_utils, data_utils

# Visualization libraries (optional)
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context='paper', style='ticks', font_scale=1.5,
        color_codes=True, rc={'legend.frameon': True})
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
FONTSIZE = 18
matplotlib.rc('xtick', labelsize=FONTSIZE) 
matplotlib.rc('ytick', labelsize=FONTSIZE) 


base_dir = './results/proteins'
proteins = ['PABP_YEAST_Fields2013', 'UBE4B_MOUSE_Klevit2013-nscor_log2_ratio', 'GFP_AEQVI_Sarkisyan2016']
#priors = ['none', 'potts', 'transformer', 'potts+transformer']
priors = ['no-unsupervised', 'potts', 'potts-only', 'transformer', 'potts+transformer', 'transformer-S', 'transformer-L']
samplers = ['PPDE']
#samplers = ['potts', 'PPDE', 'simulated_annealing', 'MALA-approx', 'CMAES', 'Random']

def n_hops(population, wt):
    K, L, V = population.shape
    nhops = []
    for k in range(K):
        prot = population[k]
        diff = ((prot - wt) > 0).astype('float')
        nhops += [diff.sum()]
    return (np.mean(nhops), np.std(nhops))
        
def diversity_score(population):
    K, L, V = population.shape
    mymap = {}    
    for k in range(K):
        
        protein = data_utils.onehot2seq( population[k][None] )[0]
        if protein in mymap:
            mymap[protein] += 1
        else:
            mymap[protein] = 1

    return len(mymap.keys())/K * 100


if __name__ == '__main__':
    wts = {}
    for protein in proteins:
        wtseqs = io_utils.read_fasta(f'./weights/{protein}/wt.fasta', return_ids=False)
        wt = data_utils.seqs_to_onehot(wtseqs)[0]
        wts[protein] = wt

    results = {}

    for p in proteins:
        results[p] = {}
        for pr in priors:
            results[p][pr] = {}
            for s in samplers:
                results[p][pr][s] = {}
                
    
                runs = glob.glob(str(Path(base_dir) / p / f'{s}_{pr}_*'))

                assert len(runs) == 1, f'Found {len(runs)} runs for {p} {s} {pr}'
                r = runs[0]

                results[p][pr][s]['log-fitness'] = np.load(os.path.join(r, 'oracle_fitness_scores.npy'))
                results[p][pr][s]['MSA Transformer score'] = np.load(os.path.join(r, 'transformer_scores.npy'))
                results[p][pr][s]['Potts score'] = np.load(os.path.join(r, 'potts_scores.npy'))
                results[p][pr][s]['population'] = np.load(os.path.join(r, 'population.npy'))
                results[p][pr][s]['energy_history'] = np.load(os.path.join(r, 'energy_history.npy'))


    for metric in ['log-fitness', 'MSA Transformer score', 'Potts score', 'diversity', 'n_hops']:
        print(metric)
        for s in samplers:
            print(s)
            for p in proteins:
                for pr in priors:
                    if metric == 'diversity':
                        if isinstance(results[p][pr][s]["population"], list):
                            pop = np.array(results[p][pr][s]["population"])[:,0]
                        else:
                            pop = results[p][pr][s]["population"]
                        print(f'[{p}]-[{pr}] diversity % = {diversity_score( pop ):.1f}')
                    elif metric == 'n_hops':
                        if isinstance(results[p][pr][s]["population"], list):
                            pop = np.array(results[p][pr][s]["population"])[:,0]
                        else:
                            pop = results[p][pr][s]["population"]
                        print(f'[{p}]-[{pr}] (mean,std) n_hops = {n_hops( pop, wts[p] )}')
                    else:
                        qs = np.quantile( results[p][pr][s][metric], [.1,.5,0.8,1.0])
                        print(f'[{p}]-[{pr}] quantiles [.1,.5,0.8,1.0] = {qs[0]:.2f},{qs[1]:.2f},{qs[2]:.2f},{qs[3]:.2f}')

        print()


    # percentiles = [0.8]
    # wts = {}
    # # comparison with baselines
    # #priors = ['potts']
    # markers = ['<', 'D', 'v', 'h']
    # markers2 = ['s', '^', 'o', '>']
    # #sampler_names = ['sim. annealing', 'MALA-approx', 'random sampling', 'PPDE (Potts only)']
    # #sampler_names2 = ['PPDE (supervised only)', 'PPDE (Potts)', 'PPDE (ESM2)', 'PPDE (Potts+ESM2)']
    # #samplers = ['simulated_annealing', 'MALA-approx', 'Random', 'potts']
    # p_names = ['PABP', 'UBE4B', 'GFP']
    # vlag = matplotlib.colormaps['flare']

    # plt.figure(figsize=(20,10))
    # for s_idx,s in enumerate(priors[:4]):
    #     plt.scatter(0,0,marker=markers[s_idx], label=sampler_names[s_idx], s=200, c='black')
    # for s_idx,s in enumerate(priors[4:]):
    #     plt.scatter(0,0,marker=markers2[s_idx], label=sampler_names2[s_idx], s=200, c='black')

    # ax=plt.gca()
    # # Shrink current axis's height by 10% on the bottom
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.3,
    #                 box.width, box.height * 0.7])

    # # Put a legend below current axis
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
    #         fancybox=True, shadow=False, ncol=4, fontsize=FONTSIZE)

    # plt.tight_layout()
    # plt.savefig('./results/proteins_main_legend.pdf')
    # #plt.show() 

    # for perc in percentiles:
    #     for p_name,p in zip(p_names,proteins):

        
    #         plt.figure(figsize=(4,4))
    #         for pr in priors:
    #             for s_idx,s in enumerate(samplers):
    #                 if isinstance(results[p][pr][s]["population"], list):
    #                     pop = np.array(results[p][pr][s]["population"])[:,0]
    #                 else:
    #                     pop = results[p][pr][s]["population"]
    #                 mean_nhops, std_nhops = n_hops( pop, wts[p] )
    #                 print(f'{samplers[s_idx]} {mean_nhops}')
    #                 lf = np.quantile( results[p][pr][s]['log-fitness'], [perc])
    #                 evo = np.quantile( results[p][pr][s]['MSA Transformer score'], [perc])
    #                 print(f'{s} {evo}')
    #                 plt.scatter(lf, evo, marker=markers[s_idx], label=sampler_names[s_idx], s=200, c=vlag( min(5,mean_nhops)/5. ))
                    
    #         for s_idx,pr in enumerate(['none', 'potts', 'transformer', 'potts+transformer']):
    #             #if pr == 'potts':
    #             #    continue
    #             if isinstance(results[p][pr]["population"], list):
    #                 pop = np.array(results[p][pr]["population"])[:,0]
    #             else:
    #                 pop = results[p][pr]["population"]
    #             mean_nhops, std_nhops = n_hops( pop, wts[p] )
    #             print(f'{sampler_names2[s_idx]} {mean_nhops}')

    #             lf = np.quantile( results[p][pr]['log-fitness'], [perc])
    #             evo = np.quantile( results[p][pr]['MSA Transformer score'], [perc])

    #             plt.scatter(lf, evo, marker=markers2[s_idx], label=sampler_names2[s_idx], s=200, c=vlag( min(5,mean_nhops)/5. ))
            
    #         plt.scatter(0.0, 0.0, s=0)
    #         plt.xlabel('log fitness', fontsize=FONTSIZE)
    #         plt.ylabel('evolutionary density', fontsize=FONTSIZE)
    #         plt.title(f'{p_name}', fontsize=FONTSIZE)
    #         plt.xticks(fontsize=FONTSIZE)
    #         plt.yticks(fontsize=FONTSIZE)
    #         plt.grid(True)
    #         #plt.tight_layout()
    #         ax=plt.gca()
    #         plt.savefig(f'./results/{p_name}_proteins_main_{perc}.pdf')
    #         #ax.legend(loc='center left', bbox_to_anchor=(1, 1.05))


    sns.set(context='notebook', style='white', rc={'legend.frameon': True, 'font.family': 'sans-serif'})

    titles = ['PABP', 'UBE4B', 'GFP']
    #samplers = ['PPDE', 'simulated_annealing', 'MALA-approx', 'CMAES', 'Random']
    #sampler_names = ['PPDE', 'sim. annealing', 'MALA-approx', 'CMA-ES', 'random sampling']
    samplers = ['PPDE']
    sampler_names = ['PPDE']
    xs = [i for i in range(0,10000,50)]
    for ti,p in zip(titles,proteins):
        df = pd.DataFrame(columns = ['sampler', 'sampler step', 'chain', '$\log p(x)$'])
        
        pr = 'potts'
        plt.figure(figsize=(6,5))
        
        for nam,s in zip(sampler_names,samplers):
            for k in range(128):
                ys = []
                if s == 'CMAES':
                    start = 1
                    upperlim = 201
                    delta = 1
                else:
                    start = 1
                    upperlim = 10001
                    delta = 50
                for j in range(start,upperlim,delta):

                    if j > start:
                        ys += [np.max(results[p][pr][s]['energy_history'][start:j,k])]
                    else:
                        ys += [results[p][pr][s]['energy_history'][start,k]]

                #plt.plot(xs, ys)# for i in range(10001)], label='random')
                df_chain = pd.DataFrame({
                    'sampler': [nam]*200,
                    'sampler step': xs,
                    'chain': [k]*200,
                    '$\log p(x)$': ys
                })
                df = df.append(df_chain, ignore_index=True)
        legend = True if p == 'PABP_YEAST_Fields2013' else False   
        sns.lineplot(data = df, x = 'sampler step', y = '$\log p(x)$', hue = 'sampler', legend = legend, linewidth=2)
        #plt.ylim(-20,8)
        plt.grid(True, axis='y')
        if legend:
            plt.legend(fontsize=FONTSIZE)
        plt.ylabel('$\log p(x)$', fontsize=FONTSIZE)
        plt.xlabel('sampler step', fontsize=FONTSIZE)
        plt.xticks(fontsize=FONTSIZE)
        plt.yticks(fontsize=FONTSIZE)
        plt.title(f'{ti}', fontsize=FONTSIZE)
        plt.savefig(f'./results/sampler_lineplot_{p}.pdf')
        plt.savefig(f'./results/sampler_lineplot_{p}.png')
        
        #plt.show()