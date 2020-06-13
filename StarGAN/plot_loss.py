import matplotlib.pyplot as plt
import pickle
import os

def plot_loss(loss_files, outfile_name):
    plot_folder = "./data/results/StarGAN/plots"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder) 


    if type(loss_files) == list:
        loss_log = []
        for file in loss_files:
            with open(f"./data/results/StarGAN/logs/{file}.pkl", "rb") as f:
                loss_log = pickle.load(f)
            plt.plot(loss_log["step"], loss_log["g_loss"])
        
        plt.xlabel("Step")
        plt.ylabel("Generator loss")
        plt.title("Loss of the generator")
        plt.savefig(f"{plot_folder}/{outfile_name}-G.png")
    else:
        with open(f"./data/results/StarGAN/logs/{loss_files}.pkl", "rb") as f:
                loss_log = pickle.load(f)

        plt.plot(loss_log["step"], loss_log["g_loss"])
        plt.xlabel("Step")
        plt.ylabel("Generator loss")
        plt.title("Loss of the generator")
        plt.savefig(f"{plot_folder}/{outfile_name}-G_loss.png")


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

if __name__ == "__main__":
    plot_loss(["loss_test", "loss_test"], "test")