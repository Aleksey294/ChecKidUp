from openai import OpenAI
import cv2
import pytesseract
import numpy as np
from sklearn.cluster import KMeans
import webcolors
import deep_translator
from PIL import Image
import streamlit as st


# Настройка пути к Tesseract-OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Словарь цветов
css3_hex_to_names = {
    '#f0f8ff': 'aliceblue', '#faebd7': 'antiquewhite', '#00ffff': 'aqua',
    '#7fffd4': 'aquamarine', '#f0ffff': 'azure', '#f5f5dc': 'beige',
    '#ffe4c4': 'bisque', '#000000': 'black', '#ffebcd': 'blanchedalmond',
    '#0000ff': 'blue', '#8a2be2': 'blueviolet', '#a52a2a': 'brown',
    '#deb887': 'burlywood', '#5f9ea0': 'cadetblue', '#7fff00': 'chartreuse',
    '#d2691e': 'chocolate', '#ff7f50': 'coral', '#6495ed': 'cornflowerblue',
    '#fff8dc': 'cornsilk', '#dc143c': 'crimson', '#00ffff': 'cyan',
    '#00008b': 'darkblue', '#008b8b': 'darkcyan', '#b8860b': 'darkgoldenrod',
    '#a9a9a9': 'darkgray', '#006400': 'darkgreen', '#bdb76b': 'darkkhaki',
    '#8b008b': 'darkmagenta', '#556b2f': 'darkolivegreen', '#ff8c00': 'darkorange',
    '#9932cc': 'darkorchid', '#8b0000': 'darkred', '#e9967a': 'darksalmon',
    '#8fbc8f': 'darkseagreen', '#483d8b': 'darkslateblue', '#2f4f4f': 'darkslategray',
    '#00ced1': 'darkturquoise', '#9400d3': 'darkviolet', '#ff1493': 'deeppink',
    '#00bfff': 'deepskyblue', '#696969': 'dimgray', '#1e90ff': 'dodgerblue',
    '#b22222': 'firebrick', '#fffaf0': 'floralwhite', '#228b22': 'forestgreen',
    '#ff00ff': 'fuchsia', '#dcdcdc': 'gainsboro', '#f8f8ff': 'ghostwhite',
    '#ffd700': 'gold', '#daa520': 'goldenrod', '#808080': 'gray', '#008000': 'green',
    '#adff2f': 'greenyellow', '#f0fff0': 'honeydew', '#ff69b4': 'hotpink',
    '#cd5c5c': 'indianred', '#4b0082': 'indigo', '#fffff0': 'ivory',
    '#f0e68c': 'khaki', '#e6e6fa': 'lavender', '#fff0f5': 'lavenderblush',
    '#7cfc00': 'lawngreen', '#fffacd': 'lemonchiffon', '#add8e6': 'lightblue',
    '#f08080': 'lightcoral', '#e0ffff': 'lightcyan', '#fafad2': 'lightgoldenrodyellow',
    '#90ee90': 'lightgreen', '#d3d3d3': 'lightgrey', '#ffb6c1': 'lightpink',
    '#ffa07a': 'lightsalmon', '#20b2aa': 'lightseagreen', '#87cefa': 'lightskyblue',
    '#778899': 'lightslategray', '#b0c4de': 'lightsteelblue', '#ffffe0': 'lightyellow',
    '#00ff00': 'lime', '#32cd32': 'limegreen', '#faf0e6': 'linen', '#ff00ff': 'magenta',
    '#800000': 'maroon', '#66cdaa': 'mediumaquamarine', '#0000cd': 'mediumblue',
    '#ba55d3': 'mediumorchid', '#9370db': 'mediumpurple', '#3cb371': 'mediumseagreen',
    '#7b68ee': 'mediumslateblue', '#00fa9a': 'mediumspringgreen', '#48d1cc': 'mediumturquoise',
    '#c71585': 'mediumvioletred', '#191970': 'midnightblue', '#f5fffa': 'mintcream',
    '#ffe4e1': 'mistyrose', '#ffe4b5': 'moccasin', '#ffdead': 'navajowhite',
    '#000080': 'navy', '#fdf5e6': 'oldlace', '#808000': 'olive', '#6b8e23': 'olivedrab',
    '#ffa500': 'orange', '#ff4500': 'orangered', '#da70d6': 'orchid', '#eee8aa': 'palegoldenrod',
    '#98fb98': 'palegreen', '#afeeee': 'paleturquoise', '#db7093': 'palevioletred',
    '#ffefd5': 'papayawhip', '#ffdab9': 'peachpuff', '#cd853f': 'peru', '#ffc0cb': 'pink',
    '#dda0dd': 'plum', '#b0e0e6': 'powderblue', '#800080': 'purple', '#663399': 'rebeccapurple',
    '#ff0000': 'red', '#bc8f8f': 'rosybrown', '#4169e1': 'royalblue', '#8b4513': 'saddlebrown',
    '#fa8072': 'salmon', '#f4a460': 'sandybrown', '#2e8b57': 'seagreen', '#fff5ee': 'seashell',
    '#a0522d': 'sienna', '#c0c0c0': 'silver', '#87ceeb': 'skyblue', '#6a5acd': 'slateblue',
    '#708090': 'slategray', '#fffafa': 'snow', '#00ff7f': 'springgreen', '#4682b4': 'steelblue',
    '#d2b48c': 'tan', '#008080': 'teal', '#d8bfd8': 'thistle', '#ff6347': 'tomato',
    '#40e0d0': 'turquoise', '#ee82ee': 'violet', '#f5deb3': 'wheat', '#ffffff': 'white',
    '#f5f5f5': 'whitesmoke', '#ffff00': 'yellow', '#9acd32': 'yellowgreen'
}


def closest_color(requested_color):
    min_colors = {}
    for hex_code, name in css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(hex_code)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]


def get_color_name(requested_color):
    try:
        closest_name = webcolors.rgb_to_name(requested_color)
    except ValueError:
        closest_name = closest_color(requested_color)
    return closest_name


def find_dominant_color(path1, k=4):
    image = cv2.imread(path1)
    image = cv2.resize(image, (100, 100))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_
    labels = kmeans.labels_
    label_counts = np.bincount(labels)
    dominant_color = colors[np.argmax(label_counts)].astype(int)
    color_name = get_color_name(tuple(dominant_color))
    return color_name


def Image_to_text(path2):
    castom_config = r'-l rus+eng --oem 3 --psm 6'
    image = cv2.imread(path2)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(binary_image, config=castom_config)
    return text

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',
)


def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    return image


st.title('ChecKidUp - сервис по оценке маркетинга и качества состава продукта детского питания')
st.markdown('Загрузите лицевую и обратную сторону упаковки продукта детского питания')
uploaded_files = st.file_uploader("", accept_multiple_files=True)
img = []
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    img.append(load_image(uploaded_file))
img.reverse()
st.image(img)
if st.button("Отправить"):
    if img == []:
        st.write("Файлы не были выбранны")
    elif len(img) > 2:
        st.write("Выбрано более двух файлов")
    elif len(img) == 1:
        st.write("Выбран только один файл")
    else:
        image1_path = 'image1.png'
        image2_path = 'image2.png'
        Image.fromarray(img[0]).save(image1_path)
        Image.fromarray(img[1]).save(image2_path)

        dominant_color = find_dominant_color(image1_path)
        nutrition_text = Image_to_text(image2_path)

        user_input = f"Проверь упаковку продукта на соответствие следующим требованиям:\n" \
                     f"1) Отсутствие доводов в пользу продукта.\n"\
                    f"2) Отсутствие доводов в пользу полезного состава продукта.\n"\
                    f"3) Отсутствие заявлений, поощряющих отказ от рекомендаций ВОЗ.\n"\
                    f"4) Отсутствие заявлений про удобство продукта.\n"\
                    f"5) Отсутствие утверждений про идеальную структуру пищи, высокое качество продукции, "\
                    f"идеальный вкус или идеальную текстуру пищи.\n"\
                    f"6) Отсутствие заявлений про пользу употребления продукта или идеализацию продукта по сравнению с домашней едой.\n"\
                    f"7) Отсутствие заявлений о наличии или отсутствии вредных или полезных ингредиентов.\n"\
                    f"8) Отсутствие заявлений с поддержкой продукта со стороны экспертов или других лиц, за исключением "\
                    f"одобрения от соответствующих национальных, региональных или международных регуляторных органов.\n"\
                    f"9) Отсутствие заявлений, касающихся благотворительности.\n"\
                    f"\n"\
                    f"Допустимо наличие информации о аллергенах и продукте в описательных выражениях (например, "\
                    f"«органическая морковь», «пшеничная мука» и т. д.), а также формулировок, относящихся к "\
                    f"религиозным требованиям или особенностям культуры (например, «вегетарианский», «халяльный»).\n"\
                    f"Продукт должен поддерживать грудное вскармливание. Ингредиенты и процентное соотношение должны "\
                    f"быть четко описаны и наименованы. Если продукт жидкий, должно быть указано предупреждение "\
                    f"не употреблять его через носик упаковки. На упаковке должна быть инструкция по приготовлению продукта.\n\n"\
                    f"Доминирующий цвет упаковки: {dominant_color}. Состав продукта: {nutrition_text}.\n"\
                    f"НАПИШИ ОТВЕТ НА РУССКОМ и выдай в ответ сначало рекомендуется или не рекомендуется, а потом уже обоснование"

        dialog_history = [{"role": "user", "content": user_input}]
        response = client.chat.completions.create(
            model="llama3:8b",
            messages=dialog_history,
        )

        response_content = deep_translator.GoogleTranslator(source='en', target='ru').translate(
            response.choices[0].message.content)
        st.write(response_content)
