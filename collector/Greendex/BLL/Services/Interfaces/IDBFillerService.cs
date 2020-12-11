using System.Threading;
using System.Threading.Tasks;

namespace BLL.Services
{
    public interface IDbFillerService
    {
        Task FillDbWithFeedContent();
    }
}