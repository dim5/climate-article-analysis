using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using Microsoft.Toolkit.Parsers.Rss;
using System.Threading.Tasks;
using Data.DTO;
using Microsoft.Extensions.Logging;

namespace BLL.Services
{
    public class RSSService : IRSSService
    {
        private const string GLOBAL_WARMING_RSS = @"https://news.google.com/rss/search?q=global+warming&hl=en-US&gl=US&ceid=US:en";
        private const string CLIMATE_RSS = @"https://news.google.com/rss/search?q=climate&hl=en-US&gl=US&ceid=US:en";

        private readonly HttpClient httpClient;
        private readonly ILogger logger;
        private static readonly RssParser rssParser = new RssParser();

        public RSSService(HttpClient httpClient, ILogger<RSSService> logger)
        {
            this.httpClient = httpClient;
            this.logger = logger;
        }

        public async Task<List<RSSResult>> GetNewsLinks(DateTime lastCheck)
        {
            logger.LogInformation("Started feed acq");
            async Task<List<RSSResult>> GetFeed(string url)
            {
                var feed = await httpClient.GetStringAsync(url);
                if (string.IsNullOrWhiteSpace(feed))
                    throw new HttpRequestException("Empty feed");
                var rss = rssParser.Parse(feed);

                return rss.Where(f => f.PublishDate > lastCheck).Select(f => new RSSResult(new Uri(f.FeedUrl), new DateTimeOffset(f.PublishDate))).ToList();
            }

            var warming = GetFeed(GLOBAL_WARMING_RSS);
            var climate = GetFeed(CLIMATE_RSS);
            var feeds = await Task.WhenAll(warming, climate);
            var res = feeds.SelectMany(f => f).Distinct().ToList();
            logger.LogInformation("Feed count {Count}", res.Count);
            return res;
        }
    }
}