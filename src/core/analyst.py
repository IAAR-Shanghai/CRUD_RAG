import csv
import json
import os
from collections import defaultdict


def save_overalls(
    output_dir: str = './output', 
    target_path: str = './statistics/overalls.csv'
) -> None:
    """"""

    # Read all evaluation results saved at output_dir
    overalls = defaultdict(lambda: defaultdict(dict))
    outputs = sorted(os.listdir(output_dir))
    for output in outputs:
        with open(os.path.join(output_dir, output)) as f:
            obj = json.load(f)
            llm, evaluator = obj['info']['llm'], obj['info']['evaluator']
            overalls[llm][evaluator] = obj['overall']
    
    # Extract table header
    evaluator_metric = []
    for obj in overalls.values():
        for evaluator, overall in obj.items():
            for metric in overall.keys():
                tmp = evaluator + ': ' + metric
                evaluator_metric.append(tmp) if tmp not in evaluator_metric else ...
    
    # Write to a csv file
    csvfile = open(target_path, 'w')
    writer = csv.writer(csvfile)
    writer.writerow(['LLM'] + evaluator_metric)
    for llm_name, obj in overalls.items():
        row = [llm_name]
        for item in evaluator_metric:
            evaluator, metric = item.split(': ')
            row.append(obj.get(evaluator, {}).get(metric, ''))
        writer.writerow(row)
    csvfile.close()

    print(f'All overalls saved at {target_path}')


def save_overalls_radar(
    overalls_path: str = './statistics/overalls.csv',
    target_path: str = './statistics/radar.pdf',
    llms: list[str] = [],
) -> None:
    """"""
    def draw_radar(data: list[list], path: str) -> None:
        import matplotlib.pyplot as plt
        import math
        plt.style.use('ggplot')

        for row in data:
            row.append(row[1])
        N = len(data[0][1:])
        angles = [i / float(N) * 2 * math.pi for i in range(N)]

        fig=plt.figure()
        ax = fig.add_subplot(111, polar=True)

        for row in data[1:]:
            ax.plot(angles, row[1:], 'o-', linewidth=2, label = row[0])
            ax.fill(angles, row[1:], alpha=0.25)

        ax.set_thetagrids([angle*180/math.pi for angle in angles], data[0][1:])
        ax.set_ylim(0, 1)
        ax.grid(True)

        plt.legend(loc='best')
        plt.savefig(path)

    f = open(overalls_path)
    csv_reader = csv.DictReader(f)
    header = [
        'LLM',
        'DiscriminativeEvaluatorKeywordLevel: avg. accuracy',
        'DiscriminativeEvaluatorSentenceLevel: avg. accuracy',
        'GenerativeEvaluator: avg. keywordsPrecision',
        'GenerativeEvaluator: avg. bertScore',
        'SelectiveEvaluator: accuracy'
    ]
    header_abbr = [
        'LLM',
        'Discri. k.w.: avg. acc.',
        'Discri. senten.: avg. acc.',
        'Gen.: avg. kwPrec',
        'Gen.: avg. bertScore',
        'Sel.: acc.'
    ]
    data = [header_abbr, ]
    for row in csv_reader:
        data.append([
            row[key] if key=='LLM' else float(row[key])
            for key in header
        ]) if row['LLM'] in llms else ...
    draw_radar(data, target_path)
    f.close()

    print(f'Radar graph saved at {target_path}')


def save_overalls_by_type(
    output_dir: str = './output', 
    evaluator_name: str = 'SelectiveEvaluator',
    metric_name: str = 'accuracy',
    target_path: str = './statistics/overalls_by_type.csv'
) -> None:
    """"""

    results = []
    filenames = [
        filename 
        for filename in os.listdir(output_dir)
        if filename.startswith(evaluator_name)
    ]
    for filename in filenames:
        f = open(os.path.join(output_dir, filename))
        obj = json.load(f)
        results.append([
            obj['info']['llm'], 
            obj['overall-doc'].get(metric_name),
            obj['overall-gen'].get(metric_name),
            obj['overall-kno'].get(metric_name),
            obj['overall-num'].get(metric_name),
        ])
        f.close()
    csvfile = open(target_path, 'w')
    writer = csv.writer(csvfile)
    writer.writerows([
        ['LLM', 'DOC', 'GEN', 'KNO', 'NUM'],
        *results
    ])

    print(f'Overalls by different types saved at {target_path}')
