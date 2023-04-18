import time
from byte_tracker import BYTETracker
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
tracker = BYTETracker()
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
            #i['bbox'] = {'x': 0.7216375470161438, 'y': 0.21243515610694885, 'w': 0.03336524963378906, 'h': 0.08091346174478531}
            # boxes.append(i['bbox']['x'])#x1
            # boxes.append(i['bbox']['y'])#y1
            # boxes.append(i['bbox']['w']+i['bbox']['x'])#x2
            # boxes.append(i['bbox']['h']+i['bbox']['y'])#y2
            print(i['label'])
            x = i['bbox']['x'] * lcd_width
            y = i['bbox']['y'] * lcd_height
            w = i['bbox']['w'] * lcd_width
            h = i['bbox']['h'] * lcd_height
            boxes.append([x,y,x+w,y+h,i['prob']])
            objlabel = i['label']
            objprob = i['prob']
            # print(boxes)
            # ui.rectangle((x,y,x+w,y+h), fill=(100,0,0,255), outline=(255,0,0,255))
            # ui.text((x,y), str(objlabel))
            # ui.text((x,y+20), str(objprob))
        boxes = np.asarray(boxes)
        if np.size(boxes) == 0:
            continue
        else:
            online_targets = tracker.update(boxes)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > 10 and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                ui.rectangle((tlwh[0],tlwh[1],tlwh[0]+tlwh[2],tlwh[1]+tlwh[3]), fill=(100,int(colours[int(tid)][0]),int(colours[int(tid)][1]),int(colours[int(tid)][2])), outline=(255,0,0,255))
                ui.text((x,y), str(tid))
            print(online_tlwhs, online_ids, online_scores)
            
            
    pipeline.config("display", (lcd_width, lcd_height, "ARGB", argb.tobytes()))
        # if tmp['nObjSize'] > 10: # try exit
        #     pipeline.free()
pipeline.free()
