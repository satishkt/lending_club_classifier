# -*- coding: utf-8 -*-
import logging
import os
from logging.config import fileConfig
import click

# from dotenv import find_dotenv, load_dotenv
from data.lc_data_janitor import LcDataExtractor


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    fileConfig('/Users/Satish/Work/lending_club_classifier/src/logging.ini')
    logger = logging.getLogger()
    logger.debug("Testing Logging")
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    lcd = LcDataExtractor()
    df = lcd.create(input_filepath)



if __name__ == '__main__':
    # log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    # project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #   load_dotenv(find_dotenv())

    main()

