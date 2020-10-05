import requests
from tqdm import tqdm
from ftplib import FTP
import os

class DataFetcher:
    def __init__(self):

        # TODO Multiple destinations for FTP
        self.datasets_info = {
            "GSE25066_RAW.tar": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE25066&format=file",
            "GSE41998_RAW.tar": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE41998&format=file",
            "GSE9782_All_MAS5_Myeloma_Variance.txt": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE9782&format=file&file=GSE9782%5FAll%5FMAS5%5FMyeloma%5FVariance%2Etxt",
            "GSE39754_RAW.tar": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE39754&format=file",
            "GSE68871_RAW.tar": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE68871&format=file",
            "GSE55145_RAW.tar": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE55145&format=file",
            "WT": {
                "url": "caftpd.nci.nih.gov",
                "destination": "/pub/OCG-DCC/TARGET/WT/mRNA-seq/L3/expression/BCCA/", 
            },
            "AML": {
                "url": "caftpd.nci.nih.gov",
                "destination": "/pub/OCG-DCC/TARGET/AML/mRNA-seq/L3/expression/BCCA/",
            },
            "ALL": {
                "url": "caftpd.nci.nih.gov",
                "destination": "/pub/OCG-DCC/TARGET/ALL/mRNA-seq/Phase1/L3/expression/BCCA/",
            },
        }

    def http_download(self, file_name, url):
        response = requests.get(url, stream=True)
        total_size_in_bytes= int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(file_name, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            return -1
        return 0

    def ftp_download(self, store_dir, url, destination):
        ftp = FTP(url)
        ftp.login()
        ftp.cwd(destination)

        os.mkdir(store_dir)

        filenames = ftp.nlst()

        for filename in filenames:
            total = ftp.size(filename)
            file = open(store_dir + "/" + filename, 'wb')

            with tqdm(total=total) as pbar:
                def callback_(data):
                    l = len(data)
                    pbar.update(l)
                    file.write(data)

                ftp.retrbinary('RETR '+ filename, callback_)

            file.close()

        ftp.quit()

        return 0

    def start_fetching(self):
        for store_data, url_data in self.datasets_info.items():
            err_code = None

            if type(url_data) == dict:
                err_code = self.ftp_download(store_data, url_data["url"], url_data["destination"])
            else:
                err_code = self.http_download(store_data, url_data)

            if err_code != 0:
                print("Error for", store_data)

if __name__ == "__main__":
    worker = DataFetcher()

    worker.start_fetching()