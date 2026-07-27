# Titanic Survival Prediction

Классическая Kaggle-задача бинарной классификации: предсказать, выжил ли пассажир Титаника, на основе демографических и билетных данных. Итоговый результат — **accuracy 0.80** на test-выборке.

## Датасет

Kaggle competition [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic). `train.csv` — для обучения, `test.csv` — для проверки модели (без cross-validation: разбиение датасета уже заложено организаторами соревнования).

## Пайплайн

### 1. Очистка данных
- Удаление дубликатов
- Анализ пропусков и распределений по числовым и категориальным признакам (train vs test сравнивались параллельно)
- `Fare == 0` помечен как пропуск (`NaN`) — нулевая цена билета физически не имеет смысла и искажала бы распределение

### 2. Feature Engineering
- **`Title`** — извлечён из поля `Name` (Mr/Mrs/Miss/…), редкие титулы сгруппированы в основные категории (`Mme`, `Lady`, `Mlle` → `Miss`; `Major`, `Col`, `Capt` → `Mr`)
- **`Fam_size`** — размер семьи на борту (`SibSp + Parch + 1`)
- **`Fam_type`** — бакетизация размера семьи (`Solo`, `Small`, `Big`, `Very big`)
- **`Ticket_len`**, **`Ticket_2letter`** — длина и префикс номера билета как прокси-признаки класса/группы бронирования
- `Sex` закодирован в числовой вид

### 3. Препроцессинг через sklearn Pipeline
- `ColumnTransformer`: числовые (`Fare`) — импутация медианой; категориальные (`Pclass`, `Title`, `Embarked`, `Fam_type`, `Ticket_len`, `Ticket_2letter`) — импутация модой + `OneHotEncoder`
- Весь препроцессинг обёрнут в единый `Pipeline` вместе с моделью — исключает утечку данных и упрощает инференс на новых данных

### 4. Модель
`RandomForestClassifier(n_estimators=500, max_depth=5, random_state=0)` — итоговый выбор после сравнения с логистической регрессией.

## Результат

**Accuracy: 0.80** на Kaggle test set.

## Стек

Python, Pandas, NumPy, scikit-learn (Pipeline, ColumnTransformer, RandomForestClassifier), Matplotlib, Seaborn

