import {
  createRouter,
  createRoute,
  createRootRoute,
  Outlet,
} from '@tanstack/react-router'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ChatPage } from './routes/index'
import { KnowledgePage } from './routes/knowledge'
import { AnalyticsPage } from './routes/analytics'

const queryClient = new QueryClient()

const rootRoute = createRootRoute({
  component: () => (
    <QueryClientProvider client={queryClient}>
      <Outlet />
    </QueryClientProvider>
  ),
})

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  component: ChatPage,
})

const knowledgeRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/knowledge',
  component: KnowledgePage,
})

const analyticsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/analytics',
  component: AnalyticsPage,
})

const routeTree = rootRoute.addChildren([indexRoute, knowledgeRoute, analyticsRoute])

export const router = createRouter({ routeTree })

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router
  }
}
