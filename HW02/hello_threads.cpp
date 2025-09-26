#include <iostream>
#include <thread>
#include <vector>
int main() {
    const int N = 5;
    std::vector<std::thread> ts; ts.reserve(N);
    for (int i = 0; i < N; ++i)
        ts.emplace_back([i,N]{ std::cout<<"Hello from thread "<<i<<" of "<<N<<"\n"; });
    for (auto &t: ts) t.join();
}
