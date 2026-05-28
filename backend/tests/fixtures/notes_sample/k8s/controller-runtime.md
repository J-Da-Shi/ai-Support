# controller-runtime

## Reconcile 循环

每个 controller 监听若干资源，事件驱动调用 Reconcile(req)。
Reconcile 应当是幂等的，返回 Result{Requeue, RequeueAfter} 决定下次执行时机。
