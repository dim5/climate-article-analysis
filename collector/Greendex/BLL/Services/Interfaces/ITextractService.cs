using System;
using System.Threading;
using System.Threading.Tasks;
using Data.DTO;

namespace BLL.Services
{
    public interface ITextractService
    {
        Task<Extract> ExtractAsync(Uri articleUrl, CancellationToken token = default);

        Task<Extract> ExtractSafeAsync(Uri articleUrl, CancellationToken token = default);
    }
}