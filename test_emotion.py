import sys, traceback
import os
# add the ai_face_persona folder (sibling to this script) to sys.path using an absolute path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "ai_face_persona")))
try:
    from emotion_model import EmotionModel
    print('imported emotion_model')
    em = EmotionModel(mode='image')
    print('created instance')
    res = em.predict([], (480,640))
    print('RESULT:', res)
except Exception as e:
    traceback.print_exc()
    print('ERROR:', e)
