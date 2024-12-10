from PIL import Image, ImageDraw
from modelscope.pipelines import pipeline
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

#文本检测
ocr_detection = pipeline(Tasks.ocr_detection, model='damo/cv_resnet18_ocr-detection-db-line-level_damo')

#文本识别
ocr_recognition = pipeline('ocr-recognition', 'damo/cv_convnextTiny_ocr-recognition-general_damo')
#NER抽取
semantic_cls = pipeline('rex-uninlu', model='damo/nlp_deberta_rex-uninlu_chinese-base')
# 信息抽取模型
schema={
        "姓名":None,
        "年龄":None
            }



#绘制涂鸦的图片
import matplotlib.pyplot as plt

# 加载原始图片（替换为实际图片路径）
image = Image.open("微信截图_20241210171557.png")  # 请替换为图片的路径
draw = ImageDraw.Draw(image)

# 多边形坐标数据
result = ocr_detection('微信截图_20241210171557.png')
polygons =result["polygons"]
# OCR 识别每个文本框

all_texts = []
text_boxes = []

for polygon in polygons:
    # 获取多边形的最小边界框
    x_coords = [polygon[i] for i in range(0, len(polygon), 2)]
    y_coords = [polygon[i] for i in range(1, len(polygon), 2)]
    bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
    
    # 裁剪图片中的该区域
    cropped_image = image.crop(bbox)
    
    # OCR 识别
    result = ocr_recognition(cropped_image)
    text = result['text'][0]
    
    # 记录文本和对应的文本框
    all_texts.append(text)
    text_boxes.append(polygon)

# 将所有文本合并成一个长字符串
combined_text = ''.join(all_texts)

# 使用 NER 模型抽取实体
entities=semantic_cls(combined_text,schema=schema)["output"]

# 涂白色背景
for entity in entities:
    entity=entity[0]
    start, end = entity['offset'][0], entity['offset'][1]
    entity_text = combined_text[start:end]
    
    # 确定实体属于哪些文本框
    cumulative_length = 0
    for i, text in enumerate(all_texts):
        cumulative_length += len(text)
        if start < cumulative_length:
            # 计算实体在文本框中的相对位置
            relative_start = start - (cumulative_length - len(text))
            relative_end = end - (cumulative_length - len(text))
            avg_char_width = (text_boxes[i][2] - text_boxes[i][0]) / len(text)
            
            # 计算具体坐标
            x_start = int(text_boxes[i][0] + relative_start * avg_char_width)
            x_end = int(text_boxes[i][0] + relative_end * avg_char_width)
            y_start = text_boxes[i][1]
            y_end = text_boxes[i][5]
            
            # 在图片上涂白色背景
            draw.rectangle([x_start, y_start, x_end, y_end], fill="white")
            break

# 使用 matplotlib 显示处理后的图片
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis("off")  # 隐藏坐标轴
plt.show()
