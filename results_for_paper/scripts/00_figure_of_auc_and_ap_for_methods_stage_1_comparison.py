import matplotlib.pyplot as plt
results_dict = {
  "dino_mc_vit": {
    "class_token_self_attention": {
      "head_3": {
        "mean_ap": 0.2006718484110813,
        "mean_auc": 0.7173344997631526
      },
      "head_4": {
        "mean_ap": 0.11001449140072203,
        "mean_auc": 0.4625681376245057
      },
      "head_0": {
        "mean_ap": 0.2127207505146175,
        "mean_auc": 0.706479919253426
      },
      "head_2": {
        "mean_ap": 0.0939133557308552,
        "mean_auc": 0.34416166350309085
      },
      "head_5": {
        "mean_ap": 0.2681803871304615,
        "mean_auc": 0.7541527601510805
      },
      "head_1": {
        "mean_ap": 0.13435654058671934,
        "mean_auc": 0.4683265370495553
      },
      "head_6": {
        "mean_ap": 0.15549944204925112,
        "mean_auc": 0.6183256887402627
      }
    },
    "value": {
      "layer_3": {
        "mean_ap": 0.20078690112249115,
        "mean_auc": 0.6861858022322281
      },
      "layer_11": {
        "mean_ap": 0.14238835341864398,
        "mean_auc": 0.6281692572399276
      },
      "layer_2": {
        "mean_ap": 0.24324714174374845,
        "mean_auc": 0.7115718430793239
      },
      "layer_1": {
        "mean_ap": 0.25640076905547116,
        "mean_auc": 0.7246030177391729
      },
      "layer_5": {
        "mean_ap": 0.1596803803797833,
        "mean_auc": 0.6329611058985273
      },
      "layer_7": {
        "mean_ap": 0.15845409971193347,
        "mean_auc": 0.6362713241263318
      },
      "layer_10": {
        "mean_ap": 0.15408017230558313,
        "mean_auc": 0.6420993348631292
      },
      "layer_0": {
        "mean_ap": 0.25460137406466843,
        "mean_auc": 0.7243955304058339
      },
      "layer_4": {
        "mean_ap": 0.1902738038456079,
        "mean_auc": 0.6729622878915873
      },
      "layer_9": {
        "mean_ap": 0.14559363773968767,
        "mean_auc": 0.625335316312893
      },
      "layer_6": {
        "mean_ap": 0.16136315070143278,
        "mean_auc": 0.6413262821347857
      },
      "layer_8": {
        "mean_ap": 0.15219871560934306,
        "mean_auc": 0.6280098528406639
      }
    },
    "query": {
      "layer_3": {
        "mean_ap": 0.22592656002360156,
        "mean_auc": 0.7056395503473917
      },
      "layer_11": {
        "mean_ap": 0.15904432985995492,
        "mean_auc": 0.6544114604481364
      },
      "layer_2": {
        "mean_ap": 0.25494433590447285,
        "mean_auc": 0.7203661809903351
      },
      "layer_1": {
        "mean_ap": 0.28004039220672916,
        "mean_auc": 0.7356198999196772
      },
      "layer_5": {
        "mean_ap": 0.18223337108255042,
        "mean_auc": 0.6622069683353037
      },
      "layer_7": {
        "mean_ap": 0.17252479710614113,
        "mean_auc": 0.65691250682491
      },
      "layer_10": {
        "mean_ap": 0.17001924410607883,
        "mean_auc": 0.6623872556465616
      },
      "layer_0": {
        "mean_ap": 0.265473827000812,
        "mean_auc": 0.7306103920846898
      },
      "layer_4": {
        "mean_ap": 0.20658906940438118,
        "mean_auc": 0.6910324205978738
      },
      "layer_9": {
        "mean_ap": 0.16467782868252254,
        "mean_auc": 0.6552889155203809
      },
      "layer_6": {
        "mean_ap": 0.17759347724579091,
        "mean_auc": 0.662465711390042
      },
      "layer_8": {
        "mean_ap": 0.1685503942190043,
        "mean_auc": 0.6570475706493205
      }
    },
    "key": {
      "layer_3": {
        "mean_ap": 0.21683237478519118,
        "mean_auc": 0.699684036383091
      },
      "layer_11": {
        "mean_ap": 0.15057367395201193,
        "mean_auc": 0.6464845113633815
      },
      "layer_2": {
        "mean_ap": 0.24880590748226236,
        "mean_auc": 0.7173765342219114
      },
      "layer_1": {
        "mean_ap": 0.25367779115262173,
        "mean_auc": 0.7251311346895963
      },
      "layer_5": {
        "mean_ap": 0.1777996671762443,
        "mean_auc": 0.6575453004066294
      },
      "layer_7": {
        "mean_ap": 0.1649443847739371,
        "mean_auc": 0.6487850242415399
      },
      "layer_10": {
        "mean_ap": 0.1650165060449457,
        "mean_auc": 0.6602495357448402
      },
      "layer_0": {
        "mean_ap": 0.2544490729560381,
        "mean_auc": 0.7252656834297804
      },
      "layer_4": {
        "mean_ap": 0.1915230331126773,
        "mean_auc": 0.680954542812701
      },
      "layer_9": {
        "mean_ap": 0.16343795053727012,
        "mean_auc": 0.653499016622451
      },
      "layer_6": {
        "mean_ap": 0.16823057316340728,
        "mean_auc": 0.6550299151078715
      },
      "layer_8": {
        "mean_ap": 0.15751075783524845,
        "mean_auc": 0.6401670248474873
      }
    }
  },

  "dino_vit": {
    "class_token_self_attention": {
      "head_3": {
        "mean_ap": 0.2609123323131377,
        "mean_auc": 0.731582631298581
      },
      "head_4": {
        "mean_ap": 0.29322375417465973,
        "mean_auc": 0.7822934060246067
      },
      "head_0": {
        "mean_ap": 0.28971697406749786,
        "mean_auc": 0.7738175650410908
      },
      "head_2": {
        "mean_ap": 0.2923874867593208,
        "mean_auc": 0.7676015095698814
      },
      "head_5": {
        "mean_ap": 0.18506273073796184,
        "mean_auc": 0.6825550234610329
      },
      "head_1": {
        "mean_ap": 0.22760218363918075,
        "mean_auc": 0.7210873554926361
      },
      "head_6": {
        "mean_ap": 0.2826472778265671,
        "mean_auc": 0.7695482481148405
      }
    },
    "value": {
      "layer_3": {
        "mean_ap": 0.2208165776551409,
        "mean_auc": 0.7219181992640911
      },
      "layer_11": {
        "mean_ap": 0.22585534689047831,
        "mean_auc": 0.728675466844254
      },
      "layer_2": {
        "mean_ap": 0.22658908563099397,
        "mean_auc": 0.7172382686918706
      },
      "layer_1": {
        "mean_ap": 0.2774456049580633,
        "mean_auc": 0.7298493191920055
      },
      "layer_5": {
        "mean_ap": 0.20038882233505445,
        "mean_auc": 0.7216012531560242
      },
      "layer_7": {
        "mean_ap": 0.2036195599439403,
        "mean_auc": 0.7163796405979741
      },
      "layer_10": {
        "mean_ap": 0.2232354493368081,
        "mean_auc": 0.7304349404137565
      },
      "layer_0": {
        "mean_ap": 0.28054946448318907,
        "mean_auc": 0.7417582171937736
      },
      "layer_4": {
        "mean_ap": 0.2117609824280564,
        "mean_auc": 0.7257350986631298
      },
      "layer_9": {
        "mean_ap": 0.2060016919575704,
        "mean_auc": 0.713436314846608
      },
      "layer_6": {
        "mean_ap": 0.21007893881299675,
        "mean_auc": 0.7224536558484742
      },
      "layer_8": {
        "mean_ap": 0.20203730114330606,
        "mean_auc": 0.7120075101925288
      }
    },
    "query": {
      "layer_3": {
        "mean_ap": 0.22138225879847898,
        "mean_auc": 0.7248835625514458
      },
      "layer_11": {
        "mean_ap": 0.21661340901268358,
        "mean_auc": 0.7244449872720046
      },
      "layer_2": {
        "mean_ap": 0.20442044210222837,
        "mean_auc": 0.7061906672539704
      },
      "layer_1": {
        "mean_ap": 0.24655120324357258,
        "mean_auc": 0.7279889549600667
      },
      "layer_5": {
        "mean_ap": 0.22140762440644754,
        "mean_auc": 0.7357898223564624
      },
      "layer_7": {
        "mean_ap": 0.21177783449827123,
        "mean_auc": 0.7275305511617052
      },
      "layer_10": {
        "mean_ap": 0.21428717680812934,
        "mean_auc": 0.7257626575756733
      },
      "layer_0": {
        "mean_ap": 0.2719974962139788,
        "mean_auc": 0.743523056148998
      },
      "layer_4": {
        "mean_ap": 0.22391727174864698,
        "mean_auc": 0.7339509807335534
      },
      "layer_9": {
        "mean_ap": 0.20008837616986894,
        "mean_auc": 0.7120291178977524
      },
      "layer_6": {
        "mean_ap": 0.220055091088364,
        "mean_auc": 0.7336328230507604
      },
      "layer_8": {
        "mean_ap": 0.20385961634979385,
        "mean_auc": 0.7188154374493498
      }
    },
    "key": {
      "layer_3": {
        "mean_ap": 0.2458667987491717,
        "mean_auc": 0.7437961578952761
      },
      "layer_11": {
        "mean_ap": 0.21986891346762893,
        "mean_auc": 0.7281189072501628
      },
      "layer_2": {
        "mean_ap": 0.22273366111459467,
        "mean_auc": 0.7254938625377134
      },
      "layer_1": {
        "mean_ap": 0.25441156075240806,
        "mean_auc": 0.732853949861143
      },
      "layer_5": {
        "mean_ap": 0.22980044173693584,
        "mean_auc": 0.7417170981639893
      },
      "layer_7": {
        "mean_ap": 0.21365796455871558,
        "mean_auc": 0.7290760206466491
      },
      "layer_10": {
        "mean_ap": 0.21692159385901247,
        "mean_auc": 0.7293104140506145
      },
      "layer_0": {
        "mean_ap": 0.2759194033634597,
        "mean_auc": 0.7366057450194924
      },
      "layer_4": {
        "mean_ap": 0.239770511550981,
        "mean_auc": 0.7462518484036258
      },
      "layer_9": {
        "mean_ap": 0.2062754209481703,
        "mean_auc": 0.7174139835416096
      },
      "layer_6": {
        "mean_ap": 0.2157608385471938,
        "mean_auc": 0.7322015730860036
      },
      "layer_8": {
        "mean_ap": 0.20644436624666757,
        "mean_auc": 0.7204028016871484
      }
    }
  },
  "dino_vit_base": {
    "class_token_self_attention": {
      "head_12": {
        "mean_ap": 0.24641258137007377,
        "mean_auc": 0.7288659681383393
      },
      "head_7": {
        "mean_ap": 0.2316406030629885,
        "mean_auc": 0.7224776679181841
      },
      "head_3": {
        "mean_ap": 0.23882465326585917,
        "mean_auc": 0.7221956535614028
      },
      "head_4": {
        "mean_ap": 0.24306581025732557,
        "mean_auc": 0.7139918864563554
      },
      "head_10": {
        "mean_ap": 0.2154048213679011,
        "mean_auc": 0.6821023738513834
      },
      "head_0": {
        "mean_ap": 0.2793113556234046,
        "mean_auc": 0.7522516146218418
      },
      "head_2": {
        "mean_ap": 0.26615665197821586,
        "mean_auc": 0.7347357031866067
      },
      "head_5": {
        "mean_ap": 0.25606055631384717,
        "mean_auc": 0.7314705619100926
      },
      "head_11": {
        "mean_ap": 0.18437034768704658,
        "mean_auc": 0.633406976986366
      },
      "head_9": {
        "mean_ap": 0.30744630957443214,
        "mean_auc": 0.7630694253314728
      },
      "head_1": {
        "mean_ap": 0.26065493827790176,
        "mean_auc": 0.7444442348581805
      },
      "head_8": {
        "mean_ap": 0.2155548837991335,
        "mean_auc": 0.7060144699035042
      },
      "head_6": {
        "mean_ap": 0.23246846218823844,
        "mean_auc": 0.7281561994329943
      }
    },
    "value": {
      "layer_11": {
        "mean_ap": 0.2332081510312643,
        "mean_auc": 0.7347622821899841
      },
      "layer_2": {
        "mean_ap": 0.21361921017313887,
        "mean_auc": 0.7077770980562312
      },
      "layer_1": {
        "mean_ap": 0.26989391544006336,
        "mean_auc": 0.7241502893514014
      },
      "layer_5": {
        "mean_ap": 0.21305211429284127,
        "mean_auc": 0.7252905778006041
      },
      "layer_7": {
        "mean_ap": 0.21943383824895168,
        "mean_auc": 0.7274766837197743
      },
      "layer_10": {
        "mean_ap": 0.23855910450809414,
        "mean_auc": 0.738716547799967
      },
      "layer_0": {
        "mean_ap": 0.27598914741491315,
        "mean_auc": 0.741576569153576
      },
      "layer_4": {
        "mean_ap": 0.2218072263531242,
        "mean_auc": 0.7326688099247285
      },
      "layer_9": {
        "mean_ap": 0.2311119343271117,
        "mean_auc": 0.7352064676058199
      },
      "layer_6": {
        "mean_ap": 0.21262057477427815,
        "mean_auc": 0.7221005267519185
      },
      "layer_8": {
        "mean_ap": 0.2225748647665086,
        "mean_auc": 0.7279903391948167
      }
    },
    "query": {
      "layer_3": {
        "mean_ap": 0.2281194804232948,
        "mean_auc": 0.7329662617413243
      },
      "layer_11": {
        "mean_ap": 0.22314495098856652,
        "mean_auc": 0.7290577934988209
      },
      "layer_2": {
        "mean_ap": 0.22343446960471844,
        "mean_auc": 0.7224014404448696
      },
      "layer_1": {
        "mean_ap": 0.26352221216443567,
        "mean_auc": 0.7377487469187547
      },
      "layer_5": {
        "mean_ap": 0.22013545022651487,
        "mean_auc": 0.7344886144688655
      },
      "layer_7": {
        "mean_ap": 0.21246413637916056,
        "mean_auc": 0.7274678544764906
      },
      "layer_10": {
        "mean_ap": 0.22533043353026666,
        "mean_auc": 0.731045669276315
      },
      "layer_0": {
        "mean_ap": 0.2653864699037066,
        "mean_auc": 0.7404436094499677
      },
      "layer_4": {
        "mean_ap": 0.2303241208060909,
        "mean_auc": 0.7420725997667377
      },
      "layer_9": {
        "mean_ap": 0.21826938651941374,
        "mean_auc": 0.7284952403555063
      },
      "layer_6": {
        "mean_ap": 0.2176037061728961,
        "mean_auc": 0.7306327937765721
      },
      "layer_8": {
        "mean_ap": 0.20789145406807905,
        "mean_auc": 0.7220516806250338
      }
    },
    "key": {
      "layer_3": {
        "mean_ap": 0.25008368357585437,
        "mean_auc": 0.749151343000248
      },
      "layer_11": {
        "mean_ap": 0.22529385368097612,
        "mean_auc": 0.7321913530761895
      },
      "layer_2": {
        "mean_ap": 0.24390957138310762,
        "mean_auc": 0.741924611846754
      },
      "layer_1": {
        "mean_ap": 0.2756622892007328,
        "mean_auc": 0.7462922904571275
      },
      "layer_5": {
        "mean_ap": 0.22767261438516087,
        "mean_auc": 0.7408431925077014
      },
      "layer_7": {
        "mean_ap": 0.2190563541506238,
        "mean_auc": 0.7319343506936473
      },
      "layer_10": {
        "mean_ap": 0.2282396510063621,
        "mean_auc": 0.7336809034727364
      },
      "layer_0": {
        "mean_ap": 0.25071837876294095,
        "mean_auc": 0.7217066696283629
      },
      "layer_4": {
        "mean_ap": 0.24021624322688676,
        "mean_auc": 0.7484456755177813
      },
      "layer_9": {
        "mean_ap": 0.2262393664469446,
        "mean_auc": 0.7345781858876758
      },
      "layer_6": {
        "mean_ap": 0.22257543128438806,
        "mean_auc": 0.7329419411425844
      },
      "layer_8": {
        "mean_ap": 0.22279436767719457,
        "mean_auc": 0.7318298336159819
      }
    }
  }
}
pixels_based_alg={
      "mean_ap": 0.29188449940164707,
      "mean_auc": 0.736533193564452
  }


plt.figure()
model_type_marker = ['*', 'o', 's']
method_types_colors = ['r', 'g', 'b', 'm']

seen_labels = set()

for (model_name, model_methods), marker_symbol in zip(results_dict.items(), model_type_marker):
    for (method_type, method_data), curr_color  in zip(model_methods.items(), method_types_colors):
        for stage, stage_data in method_data.items():
            mean_ap = stage_data["mean_ap"]
            mean_auc = stage_data["mean_auc"]
            model_label = model_name.replace("_", " ")
            method_label = method_type.replace("_", " ")
            if f"{model_label} {method_label}" not in seen_labels:
                plt.scatter(mean_ap, mean_auc, label=f"{model_label} {method_label}", marker=marker_symbol, c=curr_color)
                seen_labels.add(f"{model_label} {method_label}")
            elif model_label == 'dino vit base' and method_label == 'class token self attention' and stage == 'head_9':
                plt.scatter(mean_ap, mean_auc, label=f"{model_label} {method_label} {stage}", marker=marker_symbol,
                            c='yellow', s=100, edgecolors='black')
            else:
              plt.scatter(mean_ap, mean_auc, marker=marker_symbol, c=curr_color)
mean_ap_pixels_based = pixels_based_alg['mean_ap']
mean_auc_pixels_based = pixels_based_alg['mean_auc']
plt.scatter(mean_ap_pixels_based, mean_auc_pixels_based, label=f"pixels based alg", marker='^')
plt.legend(loc='lower right', bbox_to_anchor=(1, 0), borderaxespad=0, labelspacing=0.2)
plt.xlabel("Mean AP")
plt.ylabel("Mean AuC")
fig = plt.gcf()
# fig.set_size_inches((10, 10))
# plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
plt.grid(True)
plt.title('Stage 1 - Saliency map comparison results')
plt.tight_layout()

plt.savefig("../figures_for_paper/00_figure_of_auc_and_ap_for_methods_stage_1_comparison.pdf", format='pdf')




