using Microsoft.AspNetCore.Identity.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore;
using Data.Model;

namespace Data
{
    public class AppDbContext : IdentityDbContext
    {
        public DbSet<Article> Articles { get; set; }

        public AppDbContext(DbContextOptions<AppDbContext> options)
            : base(options)
        {
        }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            modelBuilder.Entity<Article>()
                .HasKey(c => c.Id);
            modelBuilder.Entity<Article>()
                .Property(c => c.Id)
                .ValueGeneratedOnAdd();
            modelBuilder.Entity<Article>()
                .HasIndex(c => c.Url)
                .IsUnique();
            modelBuilder.Entity<Article>()
                .HasIndex(c => c.Created);

            base.OnModelCreating(modelBuilder);
        }
    }
}