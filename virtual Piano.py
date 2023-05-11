import cv2 as cv
import numpy as np
import wave
import pyaudio
import mediapipe as mp

def play_audio(file):
    audio_chunk = 1024
    audio_wf = wave.open(file, 'rb')
    audio_p = pyaudio.PyAudio()

    audio_stream = audio_p.open(format=audio_p.get_format_from_width(audio_wf.getsampwidth()),
                    channels=audio_wf.getnchannels(),
                    rate=22050,
                    output=True)

    audio_data = audio_wf.readframes(audio_chunk)

    while audio_data:
        audio_stream.write(audio_data)
        audio_data = audio_wf.readframes(audio_chunk)

    audio_stream.stop_stream()
    audio_stream.close()

    audio_p.terminate()

def play_sound_by_index(index):
    key_sound_map = {
    0: '/Users/admin/Downloads/Virtual Piano/sound_q.wav',
    1: '/Users/admin/Downloads/Virtual Piano/sound_w.wav',
    2: '/Users/admin/Downloads/Virtual Piano/sound_e.wav',
    3: '/Users/admin/Downloads/Virtual Piano/sound_r.wav',
    4: '/Users/admin/Downloads/Virtual Piano/sound_t.wav',
    5: '/Users/admin/Downloads/Virtual Piano/sound_y.wav',
    6: '/Users/admin/Downloads/Virtual Piano/sound_u.wav',
    7: '/Users/admin/Downloads/Virtual Piano/sound_i.wav',
    8: '/Users/admin/Downloads/Virtual Piano/sound_o.wav',
    9: '/Users/admin/Downloads/Virtual Piano/sound_p.wav',
    }
    if index in key_sound_map:
        play_audio(key_sound_map[index])

# Initialize mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

finger_tips = {
    'thumb': mp_hands.HandLandmark.THUMB_TIP,
    'index': mp_hands.HandLandmark.INDEX_FINGER_TIP,
    'middle': mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    'ring': mp_hands.HandLandmark.RING_FINGER_TIP,
    'pinky': mp_hands.HandLandmark.PINKY_TIP
}

video_capture = cv.VideoCapture(0)

while True:
    _, video_frame = video_capture.read()
    video_frame = cv.resize(video_frame, (580, 600))
    video_frame = cv.flip(video_frame, 1)
    video_frame = cv.GaussianBlur(video_frame, (5, 5), 0)

    # Process frame with mediapipe hands
    rgb_frame = cv.cvtColor(video_frame, cv.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand:        
            for finger, landmark in finger_tips.items():
                # Get fingertip coordinates
                x, y = int(hand_landmarks.landmark[landmark].x * video_frame.shape[1]), \
                        int(hand_landmarks.landmark[landmark].y * video_frame.shape[0])

            # Draw circle on the fingertip
                cv.circle(video_frame, (x, y), 5, [20, 240, 255], -1)

                rect_width = 58
                for i in range(0, 580, rect_width):
                    if x > i and y > 0 and x < i + rect_width and y < 250:
                        play_sound_by_index(i // rect_width)
                        break

# Rectangle colors (B, G, R)
    color1 = (0, 255, 255)
    color2 = (0, 255, 0)


# Draw rectangles
    rect_width = 58
    for i in range(0, 580, rect_width):
        if i < 290:
            cv.rectangle(video_frame, (i, 0), (i + rect_width, 250), color1, 2)
        else:
            cv.rectangle(video_frame, (i, 0), (i + rect_width, 250), color2, 2)
    
# Display the resulting frame
    cv.imshow('Virtual Piano', video_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
video_capture.release()
cv.destroyAllWindows()
hands.close()
