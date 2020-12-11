using Hangfire.Annotations;
using Hangfire.Dashboard;

namespace Greendex.Filters
{
    public class HangDashAuth : IDashboardAuthorizationFilter
    {
        private readonly bool _isDevelopment;

        public HangDashAuth(bool isDevelopment)
        {
            _isDevelopment = isDevelopment;
        }

        public bool Authorize([NotNull] DashboardContext context)
        {
            if (_isDevelopment)
                return true;

            var httpContext = context.GetHttpContext();
            return httpContext.User.Identity.IsAuthenticated;
        }
    }
}