import sys 
from PIL 
import Image 
import lmdb 
import random 
import os 
sys.path.append('~/caffe/python/') 
import caffe 

if __name__ == '__main__': 
    k=0 i=0 in_db = lmdb.open('~/cityscape_val_lmdb',map_size=int(1e12)) 
    in_txn = in_db.begin(write=True)
    for dirs in os.listdir('~/cityscape/gtFine/train'):
        src_dir = os.walk('~/cityscape/gtFine/train/'+dirs)
        for path, dir, img_list in src_dir: 
            random.shuffle(img_list) # creating images lmdb 
            for srcimg in img_list: 
                if srcimg.endswith('_color.png'):
                    rgbimg = path + '/' + srcimg
                    labimg = path + '/' + srcimg.split('color')[0] + 'labelIds.png'
                    labname=srcimg.split('color')[0] + 'labelIds.png'
                    if os.path.exists(labimg):
                        rgb = np.array(Image.open(rgbimg))
                        Dtype = rgb.dtype rgb= rgb[:,:,::-1] 
                        rgb = Image.fromarray(rgb) 
                        rgb = np.array(rgb, Dtype) 
                        rgb = rgb.transpose((2, 0, 1)) 
                        rgb_dat = caffe.io.array_to_datum(rgb) 
                        in_txn.put(srcimg, rgb_dat.SerializeToString()) 
                        lab = np.array(Image.open(labimg), Dtype)
                        lab = Image.fromarray(lab) lab = np.array(lab, Dtype) 
                        lab = lab.reshape(lab.shape[0],lab.shape[1],1) 
                        lab = lab.transpose((2,0,1)) 
                        L_dat = caffe.io.array_to_datum(lab) 
                        in_txn.put(labname, L_dat.SerializeToString())
                        i += 1 
                        if i%100 == 0:
                            k=k+1 in_txn.commit() 
                            in_txn = in_db.begin(write=True) 
                            print 'process %d batch' % k in_txn.commit()
                            print 'process last batch!' 
    in_txn.commit()
    print 'process last batch'
    in_db.close()
    print 'finish'

