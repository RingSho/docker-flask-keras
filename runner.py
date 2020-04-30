from threading import Thread
import requests as rq
import time


MNIST1 = 'mnist1'
MNIST2 = 'mnist2'

MNIST1_END_POINT = "http://localhost:5000/predict"
MNIST2_END_POINT = "http://localhost:5001/predict"

ENGINE_DICT = {MNIST1: MNIST1_END_POINT,
               MNIST2: MNIST2_END_POINT}


def get_request(engine_name, engine_end_point):
    print("Start request: ", engine_name)
    print("End point: ", engine_end_point)
    result = rq.get(engine_end_point).json()
    print(result)
    print("Done:", engine_name)

class Runner:
    """
    各モジュールに対してthreadを作成し並列実行
    """
    def __init__(self):
        self.threads = []
        self.engine_dict = ENGINE_DICT
    
    def clear_threads(self) -> None:
        """スレッドを初期化"""
        self.threads = []

    def add_threads(self) -> None:
        """スレッドを追加"""
        self.clear_threads()

        for engine_name, engine_end_point in zip(self.engine_dict.keys(), self.engine_dict.values()):

            thread = Thread(target=get_request, args=(engine_name, engine_end_point), 
                            name=engine_name)
            thread.deamon = True
            self.threads.append(thread)

    def run_threads(self) -> None:
        """ スレッドを実行"""
        for thread in self.threads:
            thread.start()

        for thread in self.threads:
            thread.join()


if __name__ == "__main__":
    RUNNER = Runner()

    print("input 'start' or 's'")
    METHOD = input(">>>  ")

    if METHOD in ['start', 's']:
        RUNNER.add_threads()
        
        start_time = time.time()
        RUNNER.run_threads()
        print("All request time:", time.time() - start_time)