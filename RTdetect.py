import gradio as gr
import pandas as pd
import scipy.signal
import numpy as np
import tensorflow as tf
from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op 
import altair as alt
import librosa

def generate_librosa_based_features_for_clip(clip):
    #mfcc = librosa.feature.mfcc(y=clip, sr=16000, n_mfcc=40)
    sample_data = (clip/32767).astype(np.float32)
    S = librosa.feature.melspectrogram(y=sample_data, win_length=int(480), 
                                        hop_length=int(0.020*16000), n_fft=512, center=True,
                                        sr=16000, n_mels=40, fmin=125, fmax=7500, power=2)#, norm=None)

    S = librosa.power_to_db(S).squeeze()[:, 1:-1] 
    return S.T
    

infer_model = tf.lite.Interpreter(model_path="./stream_state_internal_quantize.tflite", num_threads=1)
infer_model.resize_tensor_input(0, [1,1,40], strict=True)  # initialize with fixed input size
infer_model.allocate_tensors()
input_details = infer_model.get_input_details()
output_details = infer_model.get_output_details()
print()
print("Input details:")
print(input_details)
print()
print("Output details:")
print(output_details)
print()



# Define function to process audio

detection_state = np.random.rand(100)
features_state = np.random.rand(40,100)


def process_audio_and_features(audio):
    # Resample audio to 16khz if needed
    if audio[0] != 16000:
        data = scipy.signal.resample(audio[1], int(float(audio[1].shape[0])/audio[0]*16000))
    
    data = data.astype(np.int16)
    #print(data.shape)
    res = generate_librosa_based_features_for_clip(data)
    # Get predictions
    for row in res:
        row1 = row.astype(np.int8)
        row2 = row1.reshape([40,1])
            
        features_state[:,:-1] = features_state[:,1:] 
        features_state[:,99] = row2[:,0]

        row3 = row1.reshape([1,1,40])
        infer_model.set_tensor(input_details[0]['index'], row3)
        infer_model.invoke()
        pred = infer_model.get_tensor(output_details[0]['index'])

        # Add prediction
        detection_state[:-1] = detection_state[1:] 
        detection_state[99] = pred[0,0]
        
    # Make line plot
    df = pd.DataFrame({"x": np.arange(len(detection_state)), "y": detection_state, "Model": "wakeword"})
    detectPlot = gr.LinePlot(value = df, x='x', y='y', color="Model", y_lim = (0,1), tooltip="Model",
                                width=600, height=300, x_title="Time (frames)", y_title="Model Score", color_legend_position="bottom")
         

    #Convert this grid to columnar data expected by Altair
    xm, ym = np.meshgrid(np.arange(0,100, 1), range(0, 40))
    source = pd.DataFrame({'x': xm.ravel(),
                            'y': ym.ravel(),
                            'z': features_state.ravel()})
    specPlot = alt.Chart(source).mark_rect().encode(x='x:O',y='y:O', color='z:Q')
    return specPlot, detectPlot
    #return plot,plot

# Create Gradio interface and launch

desc = """
This is a demo of RT Wake Word detection using tflite pre-trained models included in the latest release
"""

gr_int_combined_detect_spec = gr.Interface(
    title = "microWakeWord Live Demo",
    description = desc,
    css = ".flex {flex-direction: column} .gr-panel {width: 100%}",
    fn=process_audio_and_features,
    inputs=[
        gr.Audio(sources=["microphone"], type="numpy", streaming=True, show_label=True)
    ],
    outputs=[
        gr.Plot(show_label=False),
        gr.LinePlot(show_label=False)
    ],
    live=True)


gr_int_combined_detect_spec.launch(share=True)