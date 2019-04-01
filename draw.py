from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
def draw_loss():
    pdf = PdfPages("loss.pdf")

    with open('loss_log.txt','r') as f:
        figure = plt.figure()
        loss = f.readlines()[0].strip().split(" ")
        points = []
        for point in loss:
            # point = point[7:13]
            points.append(float(point))
        x = range(len(points))
        plt.plot(x,points[:len(x)])
        plt.xlabel("batch",fontsize=12)
        plt.ylabel("loss",fontsize=12)
        plt.tight_layout()
        plt.show()
        pdf.savefig(figure)
        plt.close()
        pdf.close()

def draw_train_acc():
    pdf = PdfPages("train_acc.pdf")

    with open('train_acc.txt', 'r') as f:
        figure = plt.figure()
        loss = f.readlines()[0].strip().split(" ")
        points = []
        for point in loss:
            # point = point[7:13]
            points.append(float(point))
        x = range(len(points))
        plt.plot(x, points[:len(x)])
        plt.axis([0,4,0.98,1.0])
        # plt.set_ylim(0.95,1.0)
        plt.xlabel("epoch", fontsize=12)
        plt.ylabel("accuracy", fontsize=12)
        plt.tight_layout()
        plt.show()
        pdf.savefig(figure)
        plt.close()
        pdf.close()

if __name__ == '__main__':
    draw_loss()
    draw_train_acc()