using System.Linq;
using System.Threading.Tasks;
using Data;
using Data.Model;
using FlexLabs.EntityFrameworkCore.Upsert;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;

namespace BLL.Services
{
    public class DbFillerService : IDbFillerService
    {
        private readonly AppDbContext context;
        private readonly IRSSService rssService;
        private readonly ITextractService textractService;
        private readonly ILogger logger;

        public DbFillerService(AppDbContext context, IRSSService rssService, ITextractService textractService, ILogger<DbFillerService> logger)
        {
            this.context = context;
            this.rssService = rssService;
            this.textractService = textractService;
            this.logger = logger;
        }

        public async Task FillDbWithFeedContent()
        {
            logger.LogDebug("Filling started");
            var lastSeed = await context.Articles.OrderByDescending(a => a.Created).Select(a => a.Created)
                .FirstOrDefaultAsync();

            var feed = await rssService.GetNewsLinks(lastSeed.Date);
            var jobs = feed.ToDictionary(f => f, f => textractService.ExtractSafeAsync(f.URL));
            await Task.WhenAll(jobs.Values);

            var articles = jobs.Where(kv => kv.Value.Result != null)
                .Select(kv => Article.FromExtract(kv.Value.Result, kv.Key.PublishDate)).ToList();
            try
            {
                logger.LogInformation("Number of articles textracted: {Number}", articles.Count);
                await context.Articles.UpsertRange(articles)
                    .On(a => a.Url)
                    .NoUpdate()
                    .RunAsync();
                logger.LogDebug("Filling successful");
            }
            catch (UnsupportedExpressionException exception)
            {
                logger.LogError(exception, "Filling with upsert failed, falling back");
                try
                {
                    context.Articles.AddRange(articles);
                    await context.SaveChangesAsync();
                }
                catch (DbUpdateException e)
                {
                    logger.LogCritical(e, "Filling failed real hard");
                }
            }
        }
    }
}