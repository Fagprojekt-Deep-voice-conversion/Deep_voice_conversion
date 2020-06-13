import matplotlib.pyplot as plt
import pickle
import os

def plot_loss(loss_files, outfile_name):
    plot_folder = "/work1/s183920/Deep_voice_conversion/StarGAN/logs"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder) 


    if type(loss_files) == list:
        loss_log = []
        for file in loss_files:
            with open(f"/work1/s183920/Deep_voice_conversion/StarGAN/logs/{file}.pkl", "rb") as f:
                loss_log = pickle.load(f)
            plt.plot(loss_log["step"], loss_log["g_loss"])
        
        plt.xlabel("Step")
        plt.ylabel("Generator loss")
        plt.title("Loss of the generator")
        plt.legend(["Seed 1000", "Seed 2000", "Seed 3000"])
        plt.savefig(f"{plot_folder}/{outfile_name}-G.png")
        print(f"Saving as {plot_folder}/{outfile_name}-G.png...")
    else:
        with open(f"/work1/s183920/Deep_voice_conversion/StarGAN/logs/{loss_files}.pkl", "rb") as f:
                loss_log = pickle.load(f)

        plt.plot(loss_log["step"], loss_log["g_loss"])
        plt.xlabel("Step")
        plt.ylabel("Generator loss")
        plt.title("Loss of the generator")
        plt.savefig(f"{plot_folder}/{outfile_name}-G_loss.png")
        print(f"Saving as {plot_folder}/{outfile_name}-G.png...")


    # plt.plot(loss_log["step"], loss_log["g_loss"])
    # plt.plot(loss_log["step"], loss_log["d_loss"])
    # plt.xlabel("Step")
    # plt.ylabel("Loss")
    # plt.legend(["Generator", "Discriminator"])
    # plt.savefig(f"{plot_folder}/{outfile_name}-Comparison.png")




    # plt.plot(loss_log["step"], loss_log["d_loss"])
    # plt.xlabel("Step")
    # plt.ylabel("Discriminator loss")
    # plt.savefig(f"{plot_folder}/{outfile_name}-D.png")

#if __name__ == "__main__":
plot_loss(["30min_seed1000/loss_30min_seed1000", "30min_seed2000/loss_30min_seed2000", "30min_seed3000/loss_30min_seed3000"], "30min")
