#include <iostream>
#include <thread>

void hello(size_t tid) {
  printf("hello from thread %zu\n", tid);
}

int main(int argc, char *argv[]) {
  
  if(argc != 2) {
    std::cerr << "usage: ./thread N\n";
  }

  std::vector<thread> threads;
  
  // spawn
  for(size_t i=0;i<10; ++i) {
    threads.emplace_back([i](){
      hello(i);
    });
  }
  
  // join the threads


  return 0;
}
