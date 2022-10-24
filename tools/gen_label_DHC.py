import os
import json

id = 0

if __name__ == '__main__':
    dataset_name = 'DHC'
    output_train = []
    output_val = []

    for i in range(67):
        if os.path.exists('/home/cvlab/notebooks/datadrive2/HY-DHC/annotations_new/c{:04d}.json'.format(i)):
            with open('/home/cvlab/notebooks/datadrive2/HY-DHC/annotations_new/c{:04d}.json'.format(i)) as f1:
                data = json.load(f1)
                for img, annotation in zip(data['images'], data['annotations']): # XXX
                    if id != img['id']:
                        id = img['id']
                        
                        file_name = img['file_name']
                        path = os.path.dirname(file_name)

                        nframes = img['nframes']

                        phq = annotation['phq9']
                        if phq < 10:
                            label = 1
                        # elif phq < 10:
                        #     label = 2
                        # elif phq < 20:
                        #     label = 3
                        else:
                            label = 2
                        
                        if i in [0, 2, 7, 11, 33, 56, 57]:
                            output_val.append('%s %d %d %d' % (path, nframes, label, 0))
                        else:
                            output_train.append('%s %d %d %d' % (path, nframes, label, 0))
                        
    with open('/home/cvlab/notebooks/datadrive2/HY-DHC/train_videofolder_merge.txt', 'w') as f2:
        f2.write('\n'.join(output_train))

    with open('/home/cvlab/notebooks/datadrive2/HY-DHC/val_videofolder_merge.txt', 'w') as f2:
        f2.write('\n'.join(output_val))
