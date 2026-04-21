# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ 
    Трансформация данных из Raw в Interim:
    - Унификация временных меток.
    - Линейная интерполяция пропусков API.
    - Умный клиппинг цен с учетом отрицательных значений.
    """
    logger = logging.getLogger(__name__)
    logger.info('Запуск процесса очистки: Raw -> Interim')

    try:
        # 1. Загрузка данных
        df = pd.read_csv(input_filepath)
        
        # 2. Обработка времени
        # Важно: преобразуем в Datetime и приводим к локальному времени Брюсселя
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('Europe/Brussels')
        
        # Сортировка по времени критична для создания лагов (shift) в будущем
        df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
        
        # 3. Обработка пропущенных значений
        # В энергетике пропуски обычно вызваны сбоями API. Линейная интерполяция — лучший выбор.
        initial_nans = df.isna().sum().sum()
        if initial_nans > 0:
            logger.info(f'Обнаружено {initial_nans} пропусков. Выполняется интерполяция...')
            # Интерполируем все колонки (цены соседей, погоду, нагрузку)
            df = df.interpolate(method='linear', limit_direction='both')

        # 4. Очистка выбросов (Clipping)
        # В Бельгии цены могут быть отрицательными (избыток генерации). 
        # Ограничиваем снизу на -50, чтобы модель не ловила аномальные "шумы" глубоких просадок.
        lower_bound = -50
        upper_bound = df['price'].quantile(0.995) # 0.5% самых экстремальных цен обрезаем
        
        df['price'] = df['price'].clip(lower=lower_bound, upper=upper_bound)
        
        # Аналогичный клиппинг для цен соседей (если они есть)
        for col in ['price_fr', 'price_de', 'price_nl']:
            if col in df.columns:
                df[col] = df[col].clip(lower=lower_bound, upper=df[col].quantile(0.995))

        logger.info(f'⚖️ Цена BE ограничена: [{lower_bound}, {upper_bound:.2f}]')

        # 5. Сохранение
        output_path = Path(output_filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем без индекса, timestamp остается в формате ISO с таймзоной
        df.to_csv(output_path, index=False)
        
        logger.info(f'✅ Очищенные данные сохранены: {output_filepath}')
        logger.info(f'Итоговое кол-во записей: {len(df)}')

    except Exception as e:
        logger.error(f'Ошибка в пайплайне очистки: {e}')
        raise

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    load_dotenv(find_dotenv())
    main()