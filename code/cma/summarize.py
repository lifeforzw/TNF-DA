### get final scores

import collections
import json
from pprint import pprint
from typing import List, Optional

import numpy as np
from scipy.stats import hmean
import os



def main1(
    dir_name,
):
    cur_sum = collections.defaultdict(list)
    files = os.listdir(dir_name)
    print(f'''num of edit samples is {len(files)}''')
    files.sort(key=lambda x: int(str(x).split("_")[-1].split(".")[0]))
    for case_file in files:
        case_file = os.path.join(dir_name, case_file)
        try:
            with open(case_file, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Could not decode {case_file} due to format error; skipping.")
        
        data = data['post']
        key_format = '{pre}_prompts_correct'
        for prefix in ['rewrite', 'paraphrase', 'neighborhood']:
            k = key_format.format(pre=prefix)
            cur_sum[prefix] += data[k]
    
    ans = {}
    ans['reliability'] = sum(cur_sum['rewrite']) / len(cur_sum['rewrite'])
    ans['paraphrase'] = sum(cur_sum['paraphrase']) / len(cur_sum['paraphrase'])
    ans['locality'] = sum(cur_sum['neighborhood']) / len(cur_sum['neighborhood'])
    pprint(ans)
    return

            




def main(
    dir_name,
    runs: Optional[List],
    first_n_cases=None,
    get_uncompressed=False,
    abs_path=False,
):  # runs = None -> all runs
    summaries = []
    uncompressed = []

    # for run_dir in (dir_name if not abs_path else dir_name).iterdir():
    # # Skip if we're not interested
    # if runs is not None and all(run not in str(run_dir) for run in runs):
    #     continue

    # Iterate through all case files
    cur_sum = collections.defaultdict(lambda: [])
    #files = list(dir_name.glob("*case_*.json"))
    files = os.listdir(dir_name)
    files.sort(key=lambda x: int(str(x).split("_")[-1].split(".")[0]))
    for case_file in files:
        case_file = os.path.join(dir_name, case_file)
        try:
            with open(case_file, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Could not decode {case_file} due to format error; skipping.")

        case_id = data["case_id"]
        if first_n_cases is not None and case_id >= first_n_cases:
            break

        if "time" in data:
            cur_sum["time"].append(data["time"])

        for prefix in ["pre", "post"]:
            # Probability metrics for which new should be lower (better) than true
            for key in ["rewrite_prompts_probs", "paraphrase_prompts_probs"]:
                if prefix not in data or key not in data[prefix]:
                    continue

                sum_key_discrete = f"{prefix}_{key.split('_')[0]}_success"
                sum_key_cont = f"{prefix}_{key.split('_')[0]}_diff"

                cur_sum[sum_key_discrete].append(
                    np.mean(
                        [
                            x["target_true"] > x["target_new"]
                            for x in data[prefix][key]
                        ]
                    )
                )
                cur_sum[sum_key_cont].append(
                    np.mean(
                        [
                            np.exp(-x["target_new"]) - np.exp(-x["target_true"])
                            for x in data[prefix][key]
                        ]
                    )
                )

            # Probability metrics for which true should be lower (better) than new
            sum_key_discrete = f"{prefix}_neighborhood_success"
            sum_key_cont = f"{prefix}_neighborhood_diff"
            key = "neighborhood_prompts_probs"
            if prefix in data and key in data[prefix]:
                cur_sum[sum_key_discrete].append(
                    np.mean(
                        [
                            x["target_true"] < x["target_new"]
                            for x in data[prefix][key]
                        ]
                    )
                )
                cur_sum[sum_key_cont].append(
                    np.mean(
                        [
                            np.exp(-x["target_true"]) - np.exp(-x["target_new"])
                            for x in data[prefix][key]
                        ]
                    )
                )

            # Accuracy-based evaluation metrics
            for key in ["rewrite", "paraphrase", "neighborhood"]:
                sum_key = f"{prefix}_{key}_acc"
                key = f"{key}_prompts_correct"

                if prefix not in data or key not in data[prefix]:
                    continue

                cur_sum[sum_key].append(np.mean(data[prefix][key]))

            # Generation metrics that can be directly averaged
            for key in ["ngram_entropy", "reference_score", "essence_score"]:
                if prefix in data and key in data[prefix]:
                    cur_sum[f"{prefix}_{key}"].append(data[prefix][key])

        if len(cur_sum) == 0:
            continue

    num_items = len(cur_sum[next(iter(cur_sum.keys()))])
    metadata = {
        "run_dir": str(dir_name),
        "num_cases": num_items,
    }

    uncompressed.append(dict(cur_sum, **metadata))

    cur_sum = {k: (np.mean(v), np.std(v)) for k, v in cur_sum.items()}
    for k, v in cur_sum.items():
        if all(exclude not in k for exclude in ["essence_score", "time"]):
            # Constant multiplication scales linearly with mean and stddev
            cur_sum[k] = tuple(np.around(z * 100, 2) for z in v)

    for prefix in ["pre", "post"]:
        for k_efficacy, k_generalization, k_specificity in [
            (
                f"{prefix}_rewrite_success",
                f"{prefix}_paraphrase_success",
                f"{prefix}_neighborhood_success",
            ),
            # (
            #     f"{prefix}_rewrite_acc",
            #     f"{prefix}_paraphrase_acc",
            #     f"{prefix}_neighborhood_acc",
            # ),
        ]:
            if all(k in cur_sum for k in [k_efficacy, k_generalization, k_specificity]):
                hmean_list = [
                    cur_sum[k_efficacy][0],
                    cur_sum[k_generalization][0],
                    cur_sum[k_specificity][0],
                ]

                # if f"{prefix}_ngram_entropy" in cur_sum:
                #     hmean_list.append(2 ** (cur_sum[f"{prefix}_ngram_entropy"][0] / 100))
                # if f"{prefix}_reference_score" in cur_sum:
                #     hmean_list.append(cur_sum[f"{prefix}_reference_score"][0])

                cur_sum[f"{prefix}_score"] = (hmean(hmean_list), np.nan)
                break

    cur_sum.update(metadata)
    pprint(cur_sum)
    summaries.append(cur_sum)
    pprint(summaries)
    return uncompressed if get_uncompressed else summaries


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir_name", type=str, help="Name of directory to scan for runs.",
        default='./{newt}result/{model_name}/kl{calisize}_train_results{rd}/{edsamples}corfirst{de_rate}_{interrange}_results_s{noiselevel}{tem}{sumepochs}/'
        #default='./{newt}result/{model_name}/train_results/corfirst{de_rate}_{interrange}_results_s{noiselevel}{tem}{sumepochs}/'
    )
    parser.add_argument(
        '--model_name', type=str,
        default='gpt2-xl'
    )
    parser.add_argument(
        "--runs",
        type=str,
        default=None,
        help="By default, summarizes each run in <dir_name>. "
        "If runs are specified, only evaluates those specific runs.",
    )
    parser.add_argument(
        "--first_n_cases",
        type=int,
        default=None,
        help="Restricts evaluation to first n cases in dataset. "
        "Useful for comparing different in-progress runs on the same slice of data.",
    )
    parser.add_argument(
        '--ktop',
        type=float,
        default=0.01
    )
    parser.add_argument(
        '--isabs',
        type=bool,
        default=False
    )
    parser.add_argument(
        '--edit_type',
        type=str,
        default='neuron',
        choices=['neuron', 'hidsize']
    )
    parser.add_argument(
        '--theta',
        type=float,
        default=0.0
    )
    parser.add_argument(
        '--erange',
        type=str,
        default='whole',
        choices=['whole', 'subject']
    )
    parser.add_argument(
        '--noise_level',
        default=3
    )
    parser.add_argument(
        '--de_rate',
        type=float,
        default=0.2,
    )
    parser.add_argument(
        '--template',
        type=str,
        default='t',
    )
    parser.add_argument(
        '--RN',
        type=str,
        default='',
    )
    parser.add_argument(
        '--editpos',
        type=str,
        default='all',
    )
    parser.add_argument('--calidata', action='store_true', help='if run with cali data and klloss')
    parser.add_argument('--editsamples', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--calisize', type=int, default=200)
    parser.add_argument('--runturn', type=int, default=0)
    parser.add_argument('--newt', action='store_true')
    parser.add_argument('--RDtest', action='store_true')

    args = parser.parse_args()
    rd = '_ablation' if args.RDtest else ''
    #iscalidata = 1 if args.calidata else 0
    newt = 'new_' if args.newt else ''
    k = args.ktop 
    isrn = False if args.RN == '' else True
    theta = args.theta
    alpha = args.alpha
    if k >= 1:
        k = int(k)
    str_k = str(k).replace('.', '')
    str_theta = str(theta).replace('.', '')
    str_alpha = str(alpha).replace('.', '')
    sfx = 'T' if args.isabs else 'F'
    de_rate = args.de_rate
    nlevel = args.noise_level
    tem = args.template
    input_dir = args.dir_name.format(newt=newt, edsamples=args.editsamples, calisize=args.calisize, rd=rd, de_rate=int(de_rate*100), model_name=args.model_name, \
                                      interrange=args.erange, noiselevel=nlevel, tem=tem, sumepochs=args.epochs)
    #input_dir = os.path.join(input_dir, f'''{args.edit_type}_top{str_k}{sfx}_{str_theta}{args.RN}{args.editpos[-1]}''')
    input_dir = os.path.join(input_dir, f'''{args.edit_type}_top{str_k}{sfx}_{args.RN}{args.editpos[-1]}{str_theta}{str_alpha}''')
    print(input_dir)
    main(
        input_dir,
        None if args.runs is None else args.runs.split(","),
        #args.first_n_cases,
    )
    # main1(
    #     input_dir,
    # )
