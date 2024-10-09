#!/usr/bin/env python3
import base64
import signal
import time
import os
import sys
from datetime import datetime
import time

import cv2
import numpy as np
from fastapi import Response

from nicegui import Client, app, core, run, ui

import beachbot

import traceback 




detect_timer =beachbot.utils.Timer()
read_timer = beachbot.utils.Timer()
preprocess_timer = beachbot.utils.Timer()

dataset = beachbot.ai.Dataset(beachbot.ai.Dataset.list_dataset_paths()[0])
print("Loaded", len(dataset.images), " samples from dataset")

model_paths = beachbot.ai.DerbrisDetector.list_model_paths()
print("Model paths are", model_paths)
#model_path = beachbot.get_model_path()+os.path.sep+"beachbot_yolov5s_beach-cleaning-object-detection__v3-augmented_ver__2__yolov5pytorch_1280"
model_path = beachbot.ai.DerbrisDetector.list_model_paths()[0]
print("Model path is", model_path)

model_file = model_path+os.path.sep+"best.onnx"

model_type = beachbot.ai.DerbrisDetector.get_model_type(model_path)
print("Model type is", model_type)

model_cls_list= beachbot.ai.DerbrisDetector.list_models_by_type(model_type)
print("Model classes are", model_cls_list)

model_cls = model_cls_list[0]
ai_detect = model_cls(model_file=model_file, use_accel=False)

# In case you don't have a webcam, this will provide a black placeholder image.
black_1px = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAXNSR0IArs4c6QAAAA1JREFUGFdjYGBg+A8AAQQBAHAgZQsAAAAASUVORK5CYII='
placeholder = Response(content=base64.b64decode(black_1px.encode('ascii')), media_type='image/png')

frame = cv2.imread(dataset.images[0])[..., ::-1]  # OpenCV image (BGR to RGB)
img_width = frame.shape[1]
img_height = frame.shape[0]
print("Image size is", str(img_width)+"x"+str(img_height))


print("Load AI model")
confidence_threshold=0.2
class_ids, confidences, boxes = ai_detect.apply_model(frame, confidence_threshold=confidence_threshold)
print("[ result is: ", [class_ids, confidences, boxes], "]")

print("Prepare server ...")

def sel_dataset(idx):
    global dataset, slider, sliderlabel, image
    dataset = beachbot.ai.Dataset(beachbot.ai.Dataset.list_dataset_paths()[idx.value])
    print("Loaded", len(dataset.images), " samples from dataset")
    #slider = ui.slider(min=0, max=len(dataset.images)-1, value=0, on_change=lambda x: up_img(image, val=x.value))
    slider._props['max'] = len(dataset.images)-1
    slider.update()
    slider.set_value(1)
    sliderlabel.update()
    slider.set_value(0)
    up_img(image, val=0)
    print(slider._props)
    

async def sel_model(idx, backend_id=0):
    global model_path, model_file, model_type, model_cls_list, ai_detect, slider, detect_timer
    if idx is not None:
        model_path = model_paths[idx.value]
    print("Model path is", model_path)

    model_file = model_path+os.path.sep+"best.onnx"
    model_type = beachbot.ai.DerbrisDetector.get_model_type(model_path)
    print("Model type is", model_type)

    model_cls_list= beachbot.ai.DerbrisDetector.list_models_by_type(model_type)
    print("Model classes are", model_cls_list)

    model_cls = model_cls_list[backend_id]
    #ai_detect= await run.io_bound(lambda : model_cls(model_file=model_file, use_accel=False))
    ai_detect = model_cls(model_file=model_file, use_accel=False)
    print("image", image)
    print("slider", slider)
    print("slider.get_value()", slider.value)

    up_img(image, val=slider.value)
    backend_content={key:model_cls_list[key].__name__ for key in range(len(model_cls_list))}
    sel_backend.set_options(backend_content,value=backend_id)
    detect_timer = beachbot.utils.Timer()




async def sel_backend(idx):
    await sel_model(None, backend_id=idx.value)

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def convert(frame: np.ndarray) -> bytes:
    rframe = image_resize(frame, width=1280)
    _, imencode_image = cv2.imencode('.jpg', rframe)
    return imencode_image.tobytes()



def add_imgbox(pleft=0, ptop=0, w=0, h=0, clsstr=None, color='#FF0000', align="start"):
    # color = 'SkyBlue'
    image.content += f'<rect x="{pleft*100}%" y="{ptop*100}%" ry="15" height="{h*100}%" width="{w*100}%" fill="none" stroke="{color}" stroke-width="4" />'
    if clsstr is not None:
        if align=="start":
            image.content += f'<text text-anchor="start" x="{pleft*100}%" y="{ptop*100}%" stroke="{color}" font-size="2em">{clsstr}</text>'
        else:
            image.content += f'<text text-anchor="{align}" x="{(pleft+w)*100}%" y="{(ptop+h)*100}%" stroke="{color}" font-size="2em">{clsstr}</text>'
    
def rframe(fnum=0):
    try:
        with read_timer as t:
            frame_bgr=cv2.imread(dataset.images[int(fnum)])
            frame = frame_bgr[..., ::-1]  # OpenCV image (BGR to RGB)
        confidence_threshold = slider_th.value/1000.0
        print("Detect with", confidence_threshold)
        with detect_timer as t:
            class_ids, confidences, boxes = ai_detect.apply_model(frame, confidence_threshold=confidence_threshold)
        image.content = ""
        rects = dataset.rects[int(fnum)]
        im_class_nr = [r['classid'] for r in rects]
        im_class = [dataset.classes[r['classid']] for r in rects]
        im_roi = [r['rect'] for r in rects]
        for r,c in zip(im_roi, im_class):
            add_imgbox(*r,c, color="#00FF00", align="end")
        for classid, confidence, box in zip(class_ids, confidences, boxes):
            if confidence >= 0.01:
                add_imgbox(*box, ai_detect.list_classes[classid])
        succ=True
    except Exception as x:
        print("Errror rframe:", x)
        traceback.print_exc() 
        succ=False
            

    #print(obj_res)
    return succ, frame_bgr

@app.get('/file/frame')
# Thanks to FastAPI's `app.get`` it is easy to create a web route which always provides the latest image from OpenCV.
async def grab_file_frame(fnum=0) -> Response:
    # The `video_capture.read` call is a blocking function.
    # So we run it in a separate thread (default executor) to avoid blocking the event loop.
    _, frame = await run.io_bound(lambda : rframe(fnum))
    if frame is None:
        return placeholder
    # `convert` is a CPU-intensive function, so we run it in a separate process to avoid blocking the event loop and GIL.
    jpeg = await run.cpu_bound(convert, frame)

    print("Read stat:", read_timer)
    print("Preprocess stat:", preprocess_timer)
    print("Detect stat:", detect_timer)
    update_chart()

    
    return Response(content=jpeg, media_type='image/jpeg')


async def disconnect() -> None:
    """Disconnect all clients from current running server."""
    for client_id in Client.instances:
        await core.sio.disconnect(client_id)


def handle_sigint(signum, frame) -> None:
    # `disconnect` is async, so it must be called from the event loop; we use `ui.timer` to do so.
    ui.timer(0.1, disconnect, once=True)
    # Delay the default handler to allow the disconnect to complete.
    ui.timer(1, lambda: signal.default_int_handler(signum, frame), once=True)


async def cleanup() -> None:
    # This prevents ugly stack traces when auto-reloading on code change,
    # because otherwise disconnected clients try to reconnect to the newly started server.
    await disconnect()


app.on_shutdown(cleanup)
# We also need to disconnect clients when the app is stopped with Ctrl+C,
# because otherwise they will keep requesting images which lead to unfinished subprocesses blocking the shutdown.
signal.signal(signal.SIGINT, handle_sigint)







async def handle_connection(cl : Client):
    await cl.connected()
    res = await cl.run_javascript("[window.screen.width,window.screen.height]")


def handle_start():
    dt = datetime.now()

app.on_connect(handle_connection)
app.on_startup(handle_start)

with ui.header().classes(replace='row items-center') as header:
    ui.button(on_click=lambda: left_drawer.toggle(), icon='settings').props('flat color=white')
    ui.space()
    ui.button( icon='error').props('flat color=white')

with ui.footer(value=False) as footer:
    ui.label('Beachbot robot, OIST x Community, Onna-son, Okinawa')

with ui.left_drawer().classes('bg-blue-100') as left_drawer:
    ui.label('Configure:')
    with ui.card().classes('w-full').style(f'overflow: hidden;'):
        lbl1 = ui.label('System:')
        ui.button('Shut Down', on_click=lambda: os.kill(os.getpid(), signal.SIGKILL))
        ui.separator()
        lbl2 = ui.label('Dataset:')
        data_content = {key:beachbot.ai.Dataset.list_dataset_paths()[key] for key in range(len(beachbot.ai.Dataset.list_dataset_paths()))}
        ui.select(data_content, value=0, on_change=sel_dataset)
        lbl3 = ui.label('Model:')
        model_content={key:model_paths[key].rsplit(os.path.sep)[-1] for key in range(len(model_paths))}
        ui.select(model_content, value=0, on_change=sel_model)
        lbl4 = ui.label('Backend:')
        backend_content={key:model_cls_list[key].__name__ for key in range(len(model_cls_list))}
        sel_backend=ui.select(backend_content, value=0, on_change=sel_backend)
        lbl5 = ui.label('Timing analysis:')
        chart = ui.highchart({
            'title': False,
            'chart': {'type': 'bar'},
            'xAxis': {'categories': ['Read', 'Preprocess', 'Detect']},
            'yAxis':{'title': {'text': 'seconds'}},
            'series': [
                {'name': 'Current Model', 'data': [read_timer.get_mean(), preprocess_timer.get_mean(), detect_timer.get_mean()]},
                {
                    'name': 'Current Model errorbar',
                    'type': 'errorbar',
                    #'yAxis': 1,
                    'data': [
                        [read_timer.get_mean()-read_timer.get_variance(), read_timer.get_mean()+read_timer.get_variance()],
                        [preprocess_timer.get_mean()-preprocess_timer.get_variance(), preprocess_timer.get_mean()+preprocess_timer.get_variance()],
                        [detect_timer.get_mean()-detect_timer.get_variance(), detect_timer.get_mean()+detect_timer.get_variance()]
                    ]
                },
            ],
            
        }).classes('w-full h-64')
        def update_chart():
            chart.options['series'][0]['data']=[read_timer.get_mean(), preprocess_timer.get_mean(), detect_timer.get_mean()]
            chart.options['series'][1]['data']=[
                        [read_timer.get_mean()-read_timer.get_variance(), read_timer.get_mean()+read_timer.get_variance()],
                        [preprocess_timer.get_mean()-preprocess_timer.get_variance(), preprocess_timer.get_mean()+preprocess_timer.get_variance()],
                        [detect_timer.get_mean()-detect_timer.get_variance(), detect_timer.get_mean()+detect_timer.get_variance()]
                    ]
            chart.update()

        # ui.switch("1")
        # ui.switch("2")
        # ui.switch("3")
        #ui.timer(1.0, lambda: ui.label('Tick!'), once=True)


with ui.page_sticky(position='bottom-right', x_offset=20, y_offset=20):
    ui.button(on_click=footer.toggle, icon='contact_support').props('fab')


with ui.row().classes('w-full'):
    with ui.card().style('max-width: 90%;'):
        def up_img(obj : ui.interactive_image, pleft=0, ptop=0, w=25, h=25, val=0):
            #color = 'SkyBlue'
            #color = '#FF0000' 
            #obj.content = f'<rect x="{pleft}%" y="{ptop}%" ry="15" height="{h}%" width="{w}%" fill="none" stroke="{color}" stroke-width="4" />'
            #obj.set_source(f'/video/frame?{time.time()}')
            obj.set_source(f'/file/frame?fnum={val}&t={time.time()}')
        ui.label('Model Analyzer:')
        image = ui.interactive_image(source="file/frame?fnum=0",size=(img_width,img_height)).style('width: 100%')
        slider = ui.slider(min=0, max=len(dataset.images)-1, value=0, on_change=lambda x: up_img(image, val=x.value))
        sliderlabel=ui.label().bind_text_from(slider, 'value', backward=lambda a: f'Frame {a} of {len(dataset.images)}')
        slider_th = ui.slider(min=1, max=1000, value=200, on_change=lambda x: up_img(image, val=slider.value))
        ui.label().bind_text_from(slider_th, 'value', backward=lambda a: f'Confidence threshold is {a/1000.0}')


beachbot.utils.kill_by_port(4321)
ui.run(title="Beachbot Model Analyzer", reload=False, port=4321)


