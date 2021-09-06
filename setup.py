from setuptools import find_packages, setup

setup(
    name="src",
    packages=find_packages(),
    version="0.1.0",
    description="Bias correction of air quality CAMS model"
    " predictions by using OpenAQ observations.",
    author="Antonio Pérez Velasco / Mario Santa Cruz López",
    license="MIT",
    entry_points={
        "console_scripts": [
            "extraction_openaq = src.scripts.extraction_openaq:main",
            "extraction_cams = src.scripts.extraction_cams:main",
            "transform_data = src.scripts.transform_data:main",
            "produce_data = src.scripts.produce_data:main",
            "plot_station_data = src.scripts.plotting:main_line",
            "plot_station_corrs = src.scripts.plotting:main_corrs",
            "plot_station_hourly_bias = src.scripts.plotting:main_hourly_bias",
            "plot_station_monthly_bias = src.scripts.plotting:main_monthly_bias",
            "plot_station_cdf_bias = src.scripts.plotting:main_cdf_bias",
            "model_train = src.scripts.train_model:main"
            "model_validation = src.scripts.validation:main",
        ],
    },
)
