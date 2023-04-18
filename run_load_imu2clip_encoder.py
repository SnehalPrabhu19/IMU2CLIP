import torch
from lib.imu_models import MW2StackRNNPooling

if __name__ == "__main__":

    # Generate random IMU-like motions as examples
    # imu_motions: array <n_samples x 6 x 1000>
    imu_motions = torch.rand(3, 6, 1000)

    # Load the IMU encoder
    """
        The following example .pt model is configured as
        - i2c: IMU2CLIP
        - s_i: source modality = IMU
        - t_v: target modality for alignment = Video
        - t_t: target modality for alignment = Text
        - mw2: MW2StackRNNPooling as the encoder
        - w_5.0: window size of 2.5 x 2 seconds
    """
    #path_imu_encoder = "./i2c_s_i_t_v_ie_mw2_w_5.0_master_imu_encoder.pt"
    path_imu_encoder = "./i2c_s_i_t_t_ie_mw2_w_2.5_master_imu_encoder.pt"

    loaded_imu_encoder = MW2StackRNNPooling(size_embeddings=512)
    loaded_imu_encoder.load_state_dict(torch.load(path_imu_encoder))
    loaded_imu_encoder.eval()
    print("Done loading the IMU Encoder")

    # Inference time
    imu2clip_embeddings = loaded_imu_encoder(imu_motions)
    print('Raw IMU Signals (random)', imu2clip_embeddings)
    print('Encoded IMU2CLIP embeddings', imu2clip_embeddings)
