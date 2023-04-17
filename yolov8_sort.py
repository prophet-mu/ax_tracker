import time
from kalmansort import *
import numpy as np
from ax import pipeline
from PIL import Image, ImageDraw
pipeline.load([
    'libsample_vin_ivps_joint_vo_sipy.so',
    '-p', '/home/tracker/yolov8.json',
    '-c', '2',
])
lcd_width, lcd_height = 854, 480
img = Image.new('RGBA', (lcd_width, lcd_height))
# ui = ImageDraw.ImageDraw(img)
def rgba2argb(rgba):
    r,g,b,a = rgba.split()
    return Image.merge("RGBA", (a,b,g,r))
canvas_argb = rgba2argb(img)
tracker = Sort()
colours = np.random.rand(128, 3) * 255
while pipeline.work():
    time.sleep(0.001)
    tmp = pipeline.result()
    argb = canvas_argb.copy()
    boxes = []
    if tmp and tmp['nObjSize']:
        ui = ImageDraw.ImageDraw(argb)
        for i in tmp['mObjects']:
            print(i['bbox'])
            print(i['label'])
            x = i['bbox']['x'] * lcd_width
            y = i['bbox']['y'] * lcd_height
            w = i['bbox']['w'] * lcd_width
            h = i['bbox']['h'] * lcd_height
            boxes.append([x,y,x+w,y+h,i['prob']])
            objlabel = i['label']
            objprob = i['prob']
            print(boxes)
        boxes = np.asarray(boxes)
        if np.size(boxes) == 0:
            continue
        else:
            tracks = tracker.update(boxes)
        for d in tracks:
            x1 = int(float(d[0]))
            y1 = int(float(d[1]))
            x2 = int(float(d[2]))
            y2 = int(float(d[3]))
            pred_id = str(int(d[4]))
            rgb = colours[int(d[4]) % 32]
            # pred_cls = d[5]
            
            ui.rectangle((x1,y1,x2,y2), fill=(100,int(colours[int(pred_id)][0]),int(colours[int(pred_id)][1]),int(colours[int(pred_id)][2])), outline=(255,0,0,255))
            ui.text((x,y), str(pred_id))
    pipeline.config("ui_image", (lcd_width, lcd_height, "ARGB", argb.tobytes()))
        # if tmp['nObjSize'] > 10: # try exit
        #     pipeline.free()
pipeline.free()
