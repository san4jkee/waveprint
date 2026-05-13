# Plotter Studio

Превращает фотографии в SVG-линии для пен-плоттера. Четыре режима, веб-интерфейс, оптимизация путей.

---

## Быстрый старт

```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Запустить сервер
python server.py

# 3. Открыть в браузере
# Открой index.html напрямую (двойной клик) или через:
# python -m http.server 8080  →  http://localhost:8080
```

---

## Структура проекта

```
plotter/
├── processor.py     # Ядро: 4 алгоритма обработки изображений
├── server.py        # Flask API сервер
├── index.html       # Веб-интерфейс (один файл)
├── requirements.txt
├── uploads/         # Загруженные фото (создаётся автоматически)
└── outputs/         # Готовые SVG (создаётся автоматически)
```

---

## Режимы

### contour — Контурная трассировка
Canny edge detection → polyline трассировка → упрощение RDP.
Результат: набросок карандашом, только края объектов.

Ключевые параметры:
- `blur_radius` — сглаживание шума перед Canny (нечётное число)
- `canny_low / canny_high` — пороги детектора краёв
- `simplify` — агрессивность упрощения полилиний (RDP epsilon)

### hatching — Штриховка
Параллельные линии с переменной длиной, зависящей от яркости пикселя.
Результат: гравюрный стиль, хорошо передаёт полутона.

Ключевые параметры:
- `angle` — угол линий (0=горизонтальные, 45=диагональные)
- `line_spacing` — расстояние между рядами
- `cross_hatch` — добавить второй слой под 90°
- `contour_mix` — наложить контуры поверх штриховки

### stippling — Стипплинг
Weighted rejection sampling по карте яркости → маленькие штрихи.
Результат: пунктирный, органичный стиль.

Ключевые параметры:
- `n_points` — количество точек (500–8000)
- `dot_len` — длина каждого мини-штриха (0 = точка)

### flowfield — Поле потока
Градиент изображения → поле направлений → частицы следуют по нему.
Результат: живописный, «волосяной» стиль, особенно красив на портретах.

Ключевые параметры:
- `n_lines` — количество линий-частиц
- `blur_radius` — гладкость поля (выше = более плавные кривые)
- `perpendicular` — следовать вдоль контуров (True) или поперёк (False)

---

## CLI-использование

```bash
# Contour
python processor.py photo.jpg contour --output out.svg

# Hatching с параметрами
python processor.py photo.jpg hatching \
  --params '{"angle":45,"line_spacing":8,"cross_hatch":true}' \
  --output hatching.svg

# Stippling
python processor.py photo.jpg stippling \
  --params '{"n_points":5000,"dot_len":2.5}' \
  --output stippling.svg

# Flow field
python processor.py photo.jpg flowfield \
  --params '{"n_lines":800,"blur_radius":15}' \
  --output flowfield.svg
```

---

## API endpoints

```
GET  /api/modes            — список режимов, дефолты, мета-параметры
POST /api/upload           — загрузить фото → {image_id, thumbnail}
POST /api/process          — обработать → {svg, stroke_count, elapsed_s}
GET  /api/download/<file>  — скачать SVG
GET  /api/health           — проверка сервера
```

### Пример POST /api/process
```json
{
  "image_id": "uuid-полученный-при-upload",
  "mode": "hatching",
  "params": {
    "angle": 30,
    "line_spacing": 6,
    "cross_hatch": false,
    "contour_mix": true,
    "stroke_width": 0.5
  }
}
```

---

## Следующий шаг: G-code для плоттера

После получения SVG установи vpype:
```bash
pip install vpype vpype-gcode
```

Конвертация SVG → G-code:
```bash
vpype read out.svg \
  linemerge --tolerance 0.5mm \
  linesort \
  scaleto 200mm 280mm \
  gwrite --profile generic out.gcode
```

---

## Советы по качеству

- Для **портретов**: flowfield с blur_radius=15–25 даёт лучший результат
- Для **архитектуры**: contour с низким simplify (0.5–1)
- Для **пейзажей**: hatching с angle=25–35 и contour_mix=true
- Для **абстракций**: stippling с n_points=5000+
- После генерации SVG всегда прогоняй через vpype linesort — экономит 30–60% времени печати
