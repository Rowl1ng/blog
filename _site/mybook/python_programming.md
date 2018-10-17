## 常用图像操作
判断bbox是否重合
```python
def if_in(predict, mask):
    x1, y1, w1, h1 = predict
    if mask[y1:y1+h1,x1:x1+w1, :].sum() > 0:
        return True
    return False
def if_overlap(predict, label, cutoff=.1):
    x1, y1, w1, h1 = predict
    x2, y2, w2, h2 = label
    predict_area = w1 * h1
    roi_area = w2 * h2
    dx = min(x1 + w1, x2 + w2) - max(x1, x2)
    dy = min(y1 + h1, y2 + h2) - max(y1, y2)
    if dx > 0 and dy > 0:
        inter_area = dx * dy
    else:
        return False
    return inter_area * 1.0/roi_area > cutoff or inter_area * 1.0/predict_area > cutoff
```
## 计算FROC
```python
FROC_data = np.zeros((4, len(roidb)), dtype=np.object)      
FP_summary = np.zeros((2, len(roidb)), dtype=np.object)
detection_summary = np.zeros((2, len(roidb)), dtype=np.object)
thresh = 0.1
for i, entry in enumerate(roidb):
    image_name = entry['file_name']
    mask, label = get_segm_mask(entry) 
    bboxs, segms, scores = get_predicts(image_name, bboxs_data, segms_data)
    FROC_data[0][i] = image_name
    FP_summary[0][i] = image_name
    FROC_data[0][i] = image_name
    FROC_data[1][i], FROC_data[2][i], FROC_data[3][i], detection_summary[1][i], FP_summary[1][i] = compute_FP_TP_Probs(mask, segms, scores, thresh)
total_FPs, total_sensitivity, all_probs = computeFROC(FROC_data)
```