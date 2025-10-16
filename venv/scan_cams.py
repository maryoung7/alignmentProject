import cv2

def try_open(idx, backend, name):
    cap = cv2.VideoCapture(idx, backend)
    ok = cap.isOpened()
    if ok:
        ok, frame = cap.read()
    cap.release()
    return ok and frame is not None

backends = [("MSMF", cv2.CAP_MSMF), ("DSHOW", cv2.CAP_DSHOW)]
print("Scanning indices 0..9 with MSMF and DSHOW...")
for bname, bflag in backends:
    hits = []
    for i in range(10):
        if try_open(i, bflag, bname):
            hits.append(i)
    print(f"{bname} working indices: {hits}")
