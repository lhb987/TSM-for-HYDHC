import os
import json
import argparse
import shutil
import pdb
import glob
from tqdm import tqdm

ROOT = '/131_data/DHC'
def get_phq(path):
    phq = 0
    with open(path, 'r', encoding='utf-8-sig') as f_crf:
        ann_crf = json.load(f_crf)
        phq = ann_crf['screening']['phq_9']['PHQ-9 총점']

    return phq

def get_summary(summary_path='/131_data/DHC/summary.txt'):
    table = ['male', 'female']
    crf_dir = os.path.join(ROOT, '0504data', '230504_CRF_DATA')
    summary_path = '/131_data/DHC/summary.txt'
    summary_f = open(summary_path, 'w')

    # with open(crf_dir, 'r', encoding='utf-8-sig') as f:
    #     crf = json.load(f)
    #     sex = int(crf['evaluation']['demographics']['성별']) - 1

    out_anno_dir = os.path.join(ROOT, 'data_all', 'annotations')

    for final_ann in tqdm(sorted(os.listdir(out_anno_dir))):
        sex = 'unknown'
        phq = 0
        final_ann_path = os.path.join(out_anno_dir, final_ann)
        crf_path = glob.glob(crf_dir + f'/{final_ann[:-5]}*')

        if crf_path:
            with open(crf_path[0], 'r', encoding='utf-8-sig') as f:
                crf = json.load(f)
                ind = int(crf['evaluation']['demographics']['성별']) - 1
                sex = table[ind]

        else:
            continue

        with open(final_ann_path, 'r') as f:
            data = json.load(f)
            try:
                phq = data['annotations'][0].get('phq9', 'unknown')
            except:
                print(f'Suspicious subject id: {final_ann[:-5]}')
                phq = 'unknown'

        row = f'Subject: {final_ann[:-5]} --> PHQ: {phq}, Sex: {sex}\n'
        summary_f.write(row)

    summary_f.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='data_all', help='dir name of merged data')
    parser.add_argument('--old_dir', type=str, default='data', help='dir name of old data')
    parser.add_argument('--new_dir', type=str, default='0504data', help='dir name of new data')
    parser.add_argument('--level', type=int, default=4)

    args = parser.parse_args()

    out_dir = os.path.join(ROOT, args.out_dir)
    old_dir = os.path.join(ROOT, args.old_dir)
    new_dir = os.path.join(ROOT, args.new_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    out_anno_dir = os.path.join(out_dir, 'annotations')
    if not os.path.exists(out_anno_dir):
        shutil.copytree(
            os.path.join(old_dir, 'annotations'),
            out_anno_dir
        )

    pa_dir = os.path.join(new_dir, '230504_PA_DATA')
    crf_dir = os.path.join(new_dir, '230504_CRF_DATA')

    for subject_id in tqdm(sorted(os.listdir(old_dir))):
        if subject_id in ['annotations', 'txt']:
            continue
        sub_dir = os.path.join(old_dir, subject_id)
        cmd = 'ln -s {} {}'.format(sub_dir, out_dir)
        os.system(cmd)

    return


    for subject_id in tqdm(sorted(os.listdir(pa_dir))):
        if os.path.exists(os.path.join(out_dir, subject_id)) or subject_id == 's0014':
            continue

        sub_dir = os.path.join(pa_dir, subject_id)
        # out_sub_dir = os.path.join(out_dir, subject_id)
        cmd = 'ln -s {} {}'.format(sub_dir, out_dir)
        os.system(cmd)
        
        crf_annotation = glob.glob(crf_dir + f'/{subject_id}*')[0]
        phq = get_phq(crf_annotation)

        annotations = sorted(glob.glob(sub_dir + '/*'*args.level + '.json'))
        new_annotation = {'images': [], 'annotations': []}

        for annotation in annotations:
            annotation_split = annotation.split('/')

            # e.g. int('d0007'[1:]) * 100 + int('63')
            video_id = int(annotation_split[5][1:]) * 100 + int(annotation_split[7])
            with open(annotation, 'r') as f:
                data = json.load(f)
                nframes = len(data['images'][0]['v_images'])

                for img, ann in zip(data['images'][0]['v_images'], data['annotations'][0]['v_annotations']):
                    img_dict = dict(
                        id = video_id,
                        nframes = nframes,
                        file_name = img['url'][2:]
                    )

                    ann.update({'phq9': phq})

                    new_annotation['images'].append(img_dict)
                    new_annotation['annotations'].append(ann)

        with open(os.path.join(out_anno_dir, subject_id + '.json'), 'w') as f_out:
            json.dump(new_annotation, f_out)

    


        
if __name__ == '__main__':
    # main()
    get_summary()



    
    

