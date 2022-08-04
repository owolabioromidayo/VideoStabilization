import os, sys, cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import AffineTransform

RANSAC_RESIDUAL_THRES = 3
RANSAC_MAX_TRIALS = 200

class VideoStabilizer:
    def __init__(self, VIDEO_PATH):
        self.VIDEO_PATH = VIDEO_PATH
        self.WINDOW_NAME = "Video Stabilizer"
        self.orb = cv2.ORB_create(nfeatures=400)

        self.loop()

    def loop(self):
        vid = cv2.VideoCapture(self.VIDEO_PATH)
        success = True
        prev_frame = frame = _frame = kp_prev = kp = _kp = _kp_prev = None
        
        while success:
            prev_frame = frame
            success, frame  = vid.read()
            frame = cv2.resize(frame, (500, 500))

            _kp_prev, _kp = _kp, self.orb.detect(frame,None)
            kp_prev, kp = np.copy(kp), np.array([x.pt for x in _kp]).astype(np.float32)

            if prev_frame is not None:
                #RANSAC for motion estimation
                model, inliners = ransac((kp_prev, kp),
                    AffineTransform,
                    min_samples=8,
                    residual_threshold=RANSAC_RESIDUAL_THRES,
                    max_trials=RANSAC_MAX_TRIALS)

                nkp = []
                for idx, i in enumerate(kp):
                    if inliners[idx] == True:
                        nkp.append(cv2.KeyPoint(x = i[0], y= i[1], size=4))

                frame = cv2.drawKeypoints(frame, _kp, None, color=(0,255,0), flags=0)
                frame = cv2.drawKeypoints(frame, _kp_prev, None, color=(255,0,0), flags=0)
                frame = cv2.drawKeypoints(frame, nkp, None, color=(0,0,255), flags=0)
                
                if model:
                    model = np.array(model).astype(np.float32)[:2]
                    _frame = cv2.warpAffine(prev_frame, model, dsize=(500,500))
                    _frame = cv2.resize(_frame, (500, 500))

                    cv2.imshow("Smoothed Video", _frame)

            cv2.imshow(self.WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        vid.release()
        cv2.destroyAllWindows()

def main():
    VIDEO_PATH = "video/test_freiburgdesk525.mp4"
    vs = VideoStabilizer(VIDEO_PATH)

if __name__ == "__main__":
    main()