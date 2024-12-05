# **Flood - Water Level Prediction**

## **Overview**
This project predicts water levels at gauge stations for the upcoming days using historical data from these stations and discharge station information.

## **Setup**

### **1. Clone the Repository**
    git clone https://github.com/huynhspm/Flood.git


### **2. Install Required Packages**
    cd Flood
    conda create -n flood python=3.10
    conda activate flood 
    pip install -e .

Once you installed the library, then you will be able to import it and use its functionalities.

    from Flood import calculate

Description parameters of function:

    Summary:
    Args:
        - lao_cai, vu_quang, ha_giang, bac_me, vinh_tuy, ham_yen, tuyen_quang, son_tay, ha_noi, chu, pha_lai, phu_tho
            - Water level information for measuring stations
            - Data: List[int] with a length of 28 (7 days ago, 4 timestamps per day)

        - yen_bai, hon_dau, xa_thac_ba, xa_hoa_binh, xa_tuyen_quang
            - Water level information for discharge stations:
            - Data: List[int] with a length of 36 (9 days ago, 4 timestamps per day)

        - timestamp
            - Data: List[datetime.datetime] with a length of 36 (9 days ago, 4 timestamps per day)
            - Format: (%Y-%m-%d %H:%M), including year, month, day, hour, and minute 

    Returns:
        - output:
            - Water level information for the Ha Noi measuring station
            - Data: List[int] with a length of 8 (2 upcoming days, 4 timestamps per day)

Run an example with fake data:

    import Flood
    from datetime import datetime, timedelta

    times = ['01:00', '07:00', '13:00', '19:00']
    start_date = datetime(2024, 1, 1)
    timestamp = [datetime.strptime(f'{(start_date + timedelta(days=day_offset)).strftime("%Y-%m-%d")} {time}', '%Y-%m-%d %H:%M') 
                for day_offset in range(9) for time in times]

    water_level_1 = [i for i in range(28)]
    water_level_2 = [i for i in range(36)]

    output = Flood.calculate(lao_cai=water_level_1,
                            vu_quang=water_level_1,
                            ha_giang=water_level_1,
                            bac_me=water_level_1,
                            vinh_tuy=water_level_1,
                            ham_yen=water_level_1,
                            tuyen_quang=water_level_1,
                            son_tay=water_level_1,
                            ha_noi=water_level_1,
                            chu=water_level_1,
                            pha_lai=water_level_1,
                            phu_tho=water_level_1,
                            yen_bai=water_level_2,
                            hon_dau=water_level_2,
                            xa_thac_ba=water_level_2,
                            xa_hoa_binh=water_level_2,
                            xa_tuyen_quang=water_level_2,
                            timestamp=timestamp)

    print(output)

### **3. Training**
Before starting training, make sure to configure the following environment variables:

- **`CUDA_VISIBLE_DEVICES`**: Specifies the GPU(s) to use.  
- **`WANDB_API_KEY`**: Required for Weights & Biases logging (replace `???` with your API key).  

Run the following commands to set up the environment and start training:

    export CUDA_VISIBLE_DEVICES=0
    export WANDB_API_KEY=???
    python3 src/train.py experiment=flood


### **4. Inference**

Input data must follow the format of the provided example files (`9day.xlsx`, `month.xlsx`, `year.xlsx`) in the `results` folder.

Use the following command to perform inference. The output will be saved in the `results` folder:

    python3 src/inference --input <path_to_input_file> --ckpt <path_to_checkpoint_file>

Example inference:

    python3 src/inference --input results/year.xlsx --ckpt weights/not_training_2024.ckp

## **Result**
<p align="center">
  <img src="results/result_2023.jpg" alt="Result 1" width="45%">
  <img src="results/result_2024.jpg" alt="Result 2" width="45%">
</p>