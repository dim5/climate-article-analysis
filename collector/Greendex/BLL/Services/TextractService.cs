using System;
using System.Collections.Concurrent;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using Data.DTO;
using Microsoft.Extensions.Logging;

namespace BLL.Services
{
    public class TextractService : ITextractService
    {
        private readonly HttpClient httpClient;
        private readonly ConcurrentDictionary<string, byte> blacklist;
        private readonly ILogger logger;

        public TextractService(HttpClient httpClient, ILogger<TextractService> logger)
        {
            this.httpClient = httpClient;
            this.logger = logger;
            blacklist = new ConcurrentDictionary<string, byte>();
        }

        public async Task<Extract> ExtractAsync(Uri articleUrl, CancellationToken token = default)
        {
            var res = await httpClient.GetAsync($"/extract/?url={articleUrl}", token);

            res.EnsureSuccessStatusCode();
            var resBody = await res.Content.ReadAsStringAsync();
            return Extract.FromJson(resBody);
        }

        public async Task<Extract> ExtractSafeAsync(Uri articleUrl, CancellationToken token = default)
        {
            if (blacklist.ContainsKey(articleUrl.Host))
            {
                return null;
            }
            try
            {
                return await ExtractAsync(articleUrl, token);
            }
            catch (HttpRequestException e)
            {
                logger.LogError(e, "Filling for {Host} failed", articleUrl.Host);
                blacklist.TryAdd(articleUrl.Host, 0);
                return null;
            }
        }
    }
}