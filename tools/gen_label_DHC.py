import os
import json

id = 0

if __name__ == '__main__':
    dataset_name = 'DHC'
    output_train = []
    output_val = []

    for i in range(67):
        if i==0 : i = i + 1
        if os.path.exists('/131_data/DHC/data/annotations/c{:04d}.json'.format(i)):
            with open('/131_data/DHC/data/annotations/c{:04d}.json'.format(i)) as f1:
                data = json.load(f1)
                print(data.keys())
                print(data['images'][0].keys())
                print(data['annotations'][0].keys())
                for img, annotation in zip(data['images'], data['annotations']): # XXX
                    if id != img['id']:
                        id = img['id']
                        
                        file_name = img['file_name']
                        path = os.path.dirname(file_name)

                        nframes = img['nframes']
                        phq = annotation['phq9']
                        # if phq < 5:
                        #     label = 1
                        if phq < 10:
                            label = 1
                        # elif phq < 20:
                        #     label = 3
                        else:
                            label = 2
                        
                        if i in [0, 2, 7, 11, 33, 56, 57]:
                            output_val.append('%s %d %d %d' % (path, nframes, label, 0))
                        else:
                            output_train.append('%s %d %d %d' % (path, nframes, label, 0))
                        
    with open('/131_data/DHC/train_videofolder_merge_01_23.txt', 'w') as f2:
        f2.write('\n'.join(output_train))

    with open('/131_data/DHC/val_videofolder_merge_01_23.txt', 'w') as f2:
        f2.write('\n'.join(output_val))
