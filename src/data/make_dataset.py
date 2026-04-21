# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Проводит первичную очистку данных: типы, таймзоны, выбросы.
        Raw -> Interim.
    """
    logger = logging.getLogger(__name__)
    logger.info('Запуск очистки данных: Raw -> Interim')

    # 1. Загрузка
    df = pd.read_csv(input_filepath)
    
    # 2. Приведение типов и таймзоны (Бельгия)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('Europe/Brussels')
    
    # 3. Базовая очистка выбросов (Clipping)
    # Ограничиваем цену сверху (99-й перцентиль), чтобы модель не училась на аномалиях
    upper_limit = df['price'].quantile(0.99)
    df['price'] = df['price'].clip(lower=0, upper=upper_limit)

    # 4. Сохранение в промежуточную папку (Interim)
    output_path = Path(output_filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f'Очищенные данные сохранены в {output_filepath}')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    load_dotenv(find_dotenv())
    main()