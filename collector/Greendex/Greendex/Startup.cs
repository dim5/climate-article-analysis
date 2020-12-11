using System;
using BLL.Services;
using Data;
using Greendex.Areas.Identity;
using Greendex.Filters;
using Greendex.Options;
using Hangfire;
using Microsoft.AspNetCore.Authentication.Google;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Components.Authorization;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

namespace Greendex
{
    public class Startup
    {
        public Startup(IConfiguration configuration)
        {
            Configuration = configuration;
        }

        public IConfiguration Configuration { get; }

        // This method gets called by the runtime. Use this method to add services to the container.
        // For more information on how to configure your application, visit https://go.microsoft.com/fwlink/?LinkID=398940
        public void ConfigureServices(IServiceCollection services)
        {
            services.Configure<RegistrationOptions>(Configuration.GetSection(RegistrationOptions.Registration));
            services.AddDbContextPool<AppDbContext>(options =>
                options.UseSqlServer(
                    Configuration.GetConnectionString("DefaultConnection")));

            services.AddDefaultIdentity<IdentityUser>()
                .AddEntityFrameworkStores<AppDbContext>();
            services.AddAuthentication()
                .AddGoogle(opt =>
                {
                    opt.ClientId = Configuration["Google:client_id"];
                    opt.ClientSecret = Configuration["Google:client_secret"];
                    opt.CallbackPath = "/google-signin";
                });

            services.AddHttpClient<ITextractService, TextractService>(client => client.BaseAddress = new Uri(Configuration["Textract:URL"]));
            services.AddHttpClient<IRSSService, RSSService>();
            services.AddScoped<IDbFillerService, DbFillerService>();

            services.AddRazorPages();
            services.AddServerSideBlazor();
            services.AddScoped<AuthenticationStateProvider, RevalidatingIdentityAuthenticationStateProvider<IdentityUser>>();
#if !DEBUG
            services.AddApplicationInsightsTelemetry(options => options.EnableQuickPulseMetricStream = false);
#endif
            services.AddHangfire(x => x.UseSqlServerStorage(Configuration.GetConnectionString("DefaultConnection")));
        }

        public void Configure(IApplicationBuilder app, IWebHostEnvironment env)
        {
            if (env.IsDevelopment())
            {
                app.UseDeveloperExceptionPage();
                app.UseDatabaseErrorPage();
            }
            else
            {
                app.UseExceptionHandler("/Error");
                app.UseHsts();
            }

            using (var scope = app.ApplicationServices.CreateScope())
            {
                var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();
                db.Database.Migrate();
            }

            app.UseHttpsRedirection();
            app.UseStaticFiles();

            app.UseRouting();

            app.UseAuthentication();
            app.UseAuthorization();

            app.UseHangfireServer();
            app.UseHangfireDashboard("/hangdash", new DashboardOptions
            {
                Authorization = new[] { new HangDashAuth(env.IsDevelopment()) },
                IsReadOnlyFunc = _ => !env.IsDevelopment()
            });

            RecurringJob.AddOrUpdate<IDbFillerService>(filler => filler.FillDbWithFeedContent(), Cron.Daily);

            app.UseEndpoints(endpoints =>
            {
                endpoints.MapControllers();
                endpoints.MapBlazorHub();
                endpoints.MapFallbackToPage("/_Host");
            });
        }
    }
}